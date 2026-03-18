"""
LoRA（Low-Rank Adapter）層の実装
論文 Section 2.1: Low-Rank Mixture of Experts

核心アイデア:
- 事前学習済み重み W0 は凍結
- 低ランク分解 ΔW = B @ A のみ学習（rank r << min(d, k)）
- A はガウス初期化、B はゼロ初期化 → 学習開始時 ΔW = 0
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA付き線形層
    
    h = (W0 + B @ A) @ x
    
    W0: 凍結された事前学習済み重み [d, k]
    B:  次元削減行列 [d, r]   ← ゼロ初期化
    A:  次元増加行列 [r, k]   ← ガウス初期化
    r:  LoRAランク (r << min(d, k))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        pretrained_weight: torch.Tensor = None,
        pretrained_bias: torch.Tensor = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 元の線形層（凍結）
        self.linear = nn.Linear(in_features, out_features, bias=pretrained_bias is not None)
        if pretrained_weight is not None:
            self.linear.weight.data.copy_(pretrained_weight)
        if pretrained_bias is not None:
            self.linear.bias.data.copy_(pretrained_bias)
        # W0 を凍結
        for param in self.linear.parameters():
            param.requires_grad = False

        # LoRA パラメータ（学習対象）
        # A: [rank, in_features] ← ガウス初期化
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        # B: [out_features, rank] ← ゼロ初期化
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 初期化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_features]
        return: [..., out_features]
        """
        # W0 @ x（凍結部分）
        base_out = self.linear(x)
        # (B @ A) @ x（LoRA部分）
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling

    def merge_weights(self) -> None:
        """LoRA重みを元の重みにマージ（推論高速化用）"""
        with torch.no_grad():
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling

    def get_lora_params(self):
        """LoRAパラメータのみ返す"""
        return [self.lora_A, self.lora_B]


class LoRAFFN(nn.Module):
    """
    LoRA付きFeed-Forward Network
    
    論文 Eq.(2):
    FFN_e(x) = (W_o + ΔW_o_e) · GeLU((W_i + ΔW_i_e) · x)
    
    2層の線形層（input projection + output projection）各々にLoRAを適用
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

        # Input projection: [embed_dim] → [hidden_dim] with LoRA
        self.wi = LoRALinear(
            in_features=embed_dim,
            out_features=hidden_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            pretrained_weight=pretrained_wi,
            pretrained_bias=pretrained_bi,
        )

        # Output projection: [hidden_dim] → [embed_dim] with LoRA
        self.wo = LoRALinear(
            in_features=hidden_dim,
            out_features=embed_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            pretrained_weight=pretrained_wo,
            pretrained_bias=pretrained_bo,
        )

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, embed_dim]
        return: [batch, seq_len, embed_dim]
        """
        x = self.wi(x)
        x = self.activation(x)
        x = self.wo(x)
        return x

    def get_lora_params(self):
        """LoRAパラメータのみ返す"""
        return self.wi.get_lora_params() + self.wo.get_lora_params()

    def freeze_lora(self):
        """LoRAパラメータを凍結（前ステップのエキスパートを固定する際に使用）"""
        for param in self.get_lora_params():
            param.requires_grad = False

    def unfreeze_lora(self):
        """LoRAパラメータを解凍"""
        for param in self.get_lora_params():
            param.requires_grad = True
