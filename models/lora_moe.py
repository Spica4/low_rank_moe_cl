"""
Low-Rank MoE モジュール
論文 Section 2.1: Low-Rank Mixture of Experts

MoE FFN層: 各エキスパートがLoRA FFNで構成
MoE Attention層: Q, K, V, O の各射影にLoRAを適用

論文 Eq.(5):
  h = W0·x + Σ_{t=1}^{T} B_{Et}·A_{Et}·x
  
論文 Eq.(6): Tth task の FFN
  FFN_{ET} = (W_o + Σ_{t=1}^{T-1} B^o_{Et}·A^o_{Et} [frozen] + B^o_{ET}·A^o_{ET})
             · GeLU((W_i + Σ_{t=1}^{T-1} B^i_{Et}·A^i_{Et} [frozen] + B^i_{ET}·A^i_{ET}) · x)
"""
import torch
import torch.nn as nn
from typing import List, Optional

from .lora_layers import LoRALinear, LoRAFFN


class LoRAMoEFFN(nn.Module):
    """
    Low-Rank MoE FFN層
    
    複数のLoRAエキスパートを保持し、
    ゲーティング重みに基づいてエキスパートを選択する。
    
    学習時: 現在のステップのエキスパートのみ学習可能
    テスト時: ゲーティングによるTop-1ハードルーティング
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        pretrained_wi: torch.Tensor = None,
        pretrained_wo: torch.Tensor = None,
        pretrained_bi: torch.Tensor = None,
        pretrained_bo: torch.Tensor = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.alpha = alpha

        # 元のFFN（凍結）
        self.base_wi = nn.Linear(embed_dim, hidden_dim)
        self.base_wo = nn.Linear(hidden_dim, embed_dim)
        if pretrained_wi is not None:
            self.base_wi.weight.data.copy_(pretrained_wi)
        if pretrained_wo is not None:
            self.base_wo.weight.data.copy_(pretrained_wo)
        if pretrained_bi is not None:
            self.base_wi.bias.data.copy_(pretrained_bi)
        if pretrained_bo is not None:
            self.base_wo.bias.data.copy_(pretrained_bo)
        for param in self.base_wi.parameters():
            param.requires_grad = False
        for param in self.base_wo.parameters():
            param.requires_grad = False

        self.activation = nn.GELU()

        # エキスパートのリスト（動的に追加）
        self.experts_A_i: nn.ParameterList = nn.ParameterList()  # input proj A
        self.experts_B_i: nn.ParameterList = nn.ParameterList()  # input proj B
        self.experts_A_o: nn.ParameterList = nn.ParameterList()  # output proj A
        self.experts_B_o: nn.ParameterList = nn.ParameterList()  # output proj B

        self.scaling = alpha / rank
        self.num_experts = 0

    def add_expert(self):
        """
        新しいエキスパートを追加
        論文: 新しいタスクが来るたびにエキスパートを追加
        """
        import math

        device = self.base_wi.weight.device

        # Input projection LoRA: A_i [rank, embed_dim], B_i [hidden_dim, rank]
        A_i = nn.Parameter(torch.empty(self.rank, self.embed_dim, device=device))
        B_i = nn.Parameter(torch.zeros(self.hidden_dim, self.rank, device=device))
        nn.init.kaiming_uniform_(A_i, a=math.sqrt(5))

        # Output projection LoRA: A_o [rank, hidden_dim], B_o [embed_dim, rank]
        A_o = nn.Parameter(torch.empty(self.rank, self.hidden_dim, device=device))
        B_o = nn.Parameter(torch.zeros(self.embed_dim, self.rank, device=device))
        nn.init.kaiming_uniform_(A_o, a=math.sqrt(5))

        self.experts_A_i.append(A_i)
        self.experts_B_i.append(B_i)
        self.experts_A_o.append(A_o)
        self.experts_B_o.append(B_o)
        self.num_experts += 1

        return self.num_experts - 1  # 新しいエキスパートのインデックス

    def freeze_expert(self, expert_idx: int):
        """指定エキスパートのLoRAパラメータを凍結"""
        self.experts_A_i[expert_idx].requires_grad = False
        self.experts_B_i[expert_idx].requires_grad = False
        self.experts_A_o[expert_idx].requires_grad = False
        self.experts_B_o[expert_idx].requires_grad = False

    def freeze_all_experts(self):
        """全エキスパートを凍結"""
        for idx in range(self.num_experts):
            self.freeze_expert(idx)

    def forward_single_expert(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """
        単一エキスパートでのフォワードパス（学習時に使用）
        
        論文 Eq.(6): 過去のエキスパートは凍結済みで累積、
        現在のエキスパートのみ学習可能
        
        x: [batch, seq_len, embed_dim]
        """
        # Input projection: (W_i + Σ ΔW_i) · x
        h = self.base_wi(x)  # W_i · x
        for idx in range(expert_idx + 1):
            # B_i @ A_i @ x
            lora_out = torch.einsum(
                '...d,rd->...r', x, self.experts_A_i[idx]
            )
            lora_out = torch.einsum(
                '...r,hr->...h', lora_out, self.experts_B_i[idx]
            )
            h = h + lora_out * self.scaling

        h = self.activation(h)

        # Output projection: (W_o + Σ ΔW_o) · h
        out = self.base_wo(h)  # W_o · h
        for idx in range(expert_idx + 1):
            lora_out = torch.einsum(
                '...h,rh->...r', h, self.experts_A_o[idx]
            )
            lora_out = torch.einsum(
                '...r,dr->...d', lora_out, self.experts_B_o[idx]
            )
            out = out + lora_out * self.scaling

        return out

    def forward_routed(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        ルーティングベースのフォワードパス（テスト時に使用）
        
        Top-1ハードルーティング: 各トークンが最も重みの高い
        エキスパートのみを使用
        
        x: [batch, seq_len, embed_dim]
        routing_weights: [batch, seq_len, num_experts]
        """
        # Top-1 ハードルーティング
        expert_indices = routing_weights.argmax(dim=-1)  # [batch, seq_len]
        output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            # このエキスパートに割り当てられたトークンのマスク
            mask = (expert_indices == expert_idx)  # [batch, seq_len]
            if not mask.any():
                continue

            # エキスパート出力を計算
            expert_out = self.forward_single_expert(x, expert_idx)

            # マスクされたトークンにのみ出力を割り当て
            mask_expanded = mask.unsqueeze(-1).expand_as(output)
            output = output + expert_out * mask_expanded.float()

        return output


class LoRAMoEAttention(nn.Module):
    """
    Low-Rank MoE Attention層
    
    論文 Eq.(3)-(4):
    Q, K, V, O の各射影にLoRAを適用
    
    MultiHead(Q,K,V) = Concat(head1,...,headh)(W_O + B_O·A_O)
    head_i = Attention[Q(W_Q_i + B_Q_i·A_Q_i), K(W_K_i + B_K_i·A_K_i), V(W_V_i + B_V_i·A_V_i)]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        rank: int = 8,
        alpha: float = 1.0,
        pretrained_qkv_weight: torch.Tensor = None,
        pretrained_proj_weight: torch.Tensor = None,
        pretrained_qkv_bias: torch.Tensor = None,
        pretrained_proj_bias: torch.Tensor = None,
        relative_position_bias_table: torch.Tensor = None,
        relative_position_index: torch.Tensor = None,
    ):
        """
        Swin TransformerではQKVが一つの線形層にまとまっている場合があるため、
        それに対応できるよう柔軟に設計
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rank = rank
        self.scaling = alpha / rank

        # 元のQKV射影（凍結） - Swin形式: 一つにまとまっている
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        if pretrained_qkv_weight is not None:
            self.qkv.weight.data.copy_(pretrained_qkv_weight)
        if pretrained_qkv_bias is not None:
            self.qkv.bias.data.copy_(pretrained_qkv_bias)
        for param in self.qkv.parameters():
            param.requires_grad = False

        # 元の出力射影（凍結）
        self.proj = nn.Linear(embed_dim, embed_dim)
        if pretrained_proj_weight is not None:
            self.proj.weight.data.copy_(pretrained_proj_weight)
        if pretrained_proj_bias is not None:
            self.proj.bias.data.copy_(pretrained_proj_bias)
        for param in self.proj.parameters():
            param.requires_grad = False

        # 相対位置バイアス（Swin Transformer の必須コンポーネント）
        # これがないと attention logit が大きくなり fp16 で NaN overflow が発生する
        if relative_position_bias_table is not None:
            self.register_buffer(
                "relative_position_bias_table", relative_position_bias_table
            )
            self.register_buffer(
                "relative_position_index", relative_position_index
            )
        else:
            self.relative_position_bias_table = None
            self.relative_position_index = None

        # エキスパートLoRAパラメータ
        # QKV用: A [rank, embed_dim], B [3*embed_dim, rank]
        self.experts_A_qkv: nn.ParameterList = nn.ParameterList()
        self.experts_B_qkv: nn.ParameterList = nn.ParameterList()
        # Proj用: A [rank, embed_dim], B [embed_dim, rank]
        self.experts_A_proj: nn.ParameterList = nn.ParameterList()
        self.experts_B_proj: nn.ParameterList = nn.ParameterList()

        self.num_experts = 0

    def add_expert(self):
        """新しいエキスパートを追加"""
        import math

        device = self.qkv.weight.device

        A_qkv = nn.Parameter(torch.empty(self.rank, self.embed_dim, device=device))
        B_qkv = nn.Parameter(torch.zeros(self.embed_dim * 3, self.rank, device=device))
        nn.init.kaiming_uniform_(A_qkv, a=math.sqrt(5))

        A_proj = nn.Parameter(torch.empty(self.rank, self.embed_dim, device=device))
        B_proj = nn.Parameter(torch.zeros(self.embed_dim, self.rank, device=device))
        nn.init.kaiming_uniform_(A_proj, a=math.sqrt(5))

        self.experts_A_qkv.append(A_qkv)
        self.experts_B_qkv.append(B_qkv)
        self.experts_A_proj.append(A_proj)
        self.experts_B_proj.append(B_proj)
        self.num_experts += 1

        return self.num_experts - 1

    def freeze_expert(self, expert_idx: int):
        self.experts_A_qkv[expert_idx].requires_grad = False
        self.experts_B_qkv[expert_idx].requires_grad = False
        self.experts_A_proj[expert_idx].requires_grad = False
        self.experts_B_proj[expert_idx].requires_grad = False

    def freeze_all_experts(self):
        for idx in range(self.num_experts):
            self.freeze_expert(idx)

    def _compute_lora_qkv(self, x: torch.Tensor, up_to_expert: int) -> torch.Tensor:
        """累積LoRA QKV出力を計算"""
        lora_out = torch.zeros(
            *x.shape[:-1], self.embed_dim * 3, device=x.device, dtype=x.dtype
        )
        for idx in range(up_to_expert + 1):
            h = torch.einsum('...d,rd->...r', x, self.experts_A_qkv[idx])
            h = torch.einsum('...r,or->...o', h, self.experts_B_qkv[idx])
            lora_out = lora_out + h * self.scaling
        return lora_out

    def _compute_lora_proj(self, x: torch.Tensor, up_to_expert: int) -> torch.Tensor:
        """累積LoRA Proj出力を計算"""
        lora_out = torch.zeros(
            *x.shape[:-1], self.embed_dim, device=x.device, dtype=x.dtype
        )
        for idx in range(up_to_expert + 1):
            h = torch.einsum('...d,rd->...r', x, self.experts_A_proj[idx])
            h = torch.einsum('...r,or->...o', h, self.experts_B_proj[idx])
            lora_out = lora_out + h * self.scaling
        return lora_out

    def forward_single_expert(
        self,
        x: torch.Tensor,
        expert_idx: int,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        単一エキスパートでのAttention計算
        
        x: [batch, seq_len, embed_dim]
        """
        B, N, C = x.shape

        # QKV計算
        qkv = self.qkv(x) + self._compute_lora_qkv(x, expert_idx)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        # 相対位置バイアスを加算（Swin Transformer に必須）
        # これを省くと attention logit が大きくなり fp16 で nan overflow が起きる
        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)
            ].reshape(N, N, -1).permute(2, 0, 1).contiguous()  # [nH, N, N]
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # MONAI shifted-window mask: (nW, N, N)
            # attn: (B*nW, num_heads, N, N) → reshape して加算
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        out = self.proj(out) + self._compute_lora_proj(out, expert_idx)

        return out
