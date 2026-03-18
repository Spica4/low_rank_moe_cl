"""
Swin-UNETR + Low-Rank MoE 統合モデル
論文: 3D Swin Transformerを拡張し、MoE構造を組み込む

MONAIのSwinUNETRをベースに:
1. Swin Transformer Blockの FFN層 と Attention層 にLoRA MoEを挿入
2. 言語ガイド付きゲーティングでエキスパートを選択
3. 各ステップでエキスパートを追加・凍結
"""
import weakref

import torch
import torch.nn as nn
from typing import List, Dict

from .lora_moe import LoRAMoEFFN, LoRAMoEAttention
from .language_gating import LanguageGuidedGating, CLIPTextEncoder


class _MoEFFNWrapper(nn.Module):
    """
    SwinTransformerBlock.mlp をMoE FFNに差し替えるラッパー。
    block.mlp(x) → moe_ffn.forward_single_expert(x, expert_idx)
    """

    def __init__(self, moe_ffn: LoRAMoEFFN, model_ref: "SwinUNETRMoE"):
        super().__init__()
        self.moe_ffn = moe_ffn
        # weakref を使い PyTorch のモジュールツリーに循環参照を作らない
        self._model_ref = weakref.ref(model_ref)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.moe_ffn.num_experts == 0:
            # エキスパート未追加時はベース重みのみで計算
            h = self.moe_ffn.activation(self.moe_ffn.base_wi(x))
            return self.moe_ffn.base_wo(h)
        expert_idx = self._model_ref()._current_expert_idx
        return self.moe_ffn.forward_single_expert(x, expert_idx)


class _MoEAttnWrapper(nn.Module):
    """
    SwinTransformerBlock.attn をMoE Attentionに差し替えるラッパー。
    MONAI の WindowAttention は (out, attn_weights) を返すため同じ I/F に合わせる。
    入力 x は window-partitioned: (num_windows*B, window_size^3, C)
    """

    def __init__(self, moe_attn: LoRAMoEAttention, model_ref: "SwinUNETRMoE"):
        super().__init__()
        self.moe_attn = moe_attn
        # weakref を使い PyTorch のモジュールツリーに循環参照を作らない
        self._model_ref = weakref.ref(model_ref)

    def forward(self, x: torch.Tensor, mask=None) -> tuple:
        if self.moe_attn.num_experts == 0:
            # エキスパート未追加時はベース重みのみで計算
            B, N, C = x.shape
            qkv = self.moe_attn.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.moe_attn.num_heads, self.moe_attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = self.moe_attn.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            out = self.moe_attn.proj(out)
            return out, None
        expert_idx = self._model_ref()._current_expert_idx
        out = self.moe_attn.forward_single_expert(x, expert_idx, mask=mask)
        return out, None


class SwinUNETRMoE(nn.Module):
    """
    Swin-UNETR + Low-Rank MoE

    MONAIのSwinUNETRをベースに、各Swin Transformer Blockの
    FFNとAttentionにLoRA MoEエキスパートを挿入する。

    使い方:
        1. model = SwinUNETRMoE(config) で初期化
        2. model.prepare_step(step=1, text_desc="...", num_classes=14)
        3. Step1の学習
        4. model.prepare_step(step=2, text_desc="...", num_classes=15)
        5. Step2の学習（Step1のエキスパートは自動凍結）
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_step = 0
        self.text_embeddings: List[torch.Tensor] = []

        # forward時に各ラッパーが参照するエキスパートインデックス
        self._current_expert_idx: int = 0

        # CLIPテキストエンコーダ
        self.clip_encoder = CLIPTextEncoder(config.model.clip_model_name)

        # ベースモデル: MONAI SwinUNETR
        self._build_base_model()

        # MoEモジュールを格納
        self.moe_ffn_layers: nn.ModuleList = nn.ModuleList()
        self.moe_attn_layers: nn.ModuleList = nn.ModuleList()

        # 言語ガイドゲーティング
        self.gating_modules: nn.ModuleDict = nn.ModuleDict()

        # MoEを作成し、ブロックのmlp/attnを差し替え
        self._inject_moe_layers()

    def _build_base_model(self):
        """MONAIのSwinUNETRを構築"""
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError:
            raise ImportError("MONAIが必要です: pip install monai")

        self.base_model = SwinUNETR(
            img_size=self.config.model.img_size,
            in_channels=self.config.model.in_channels,
            out_channels=self.config.data.step1_num_classes,
            feature_size=self.config.model.feature_size,
            use_checkpoint=False,
        )

        # ベースモデルのパラメータを凍結
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _inject_moe_layers(self):
        """
        Swin Transformer BlockのFFNとAttentionにMoEを挿入し、
        block.mlp / block.attn をラッパーで差し替える。
        """
        swin_vit = self.base_model.swinViT

        all_layer_groups = [
            swin_vit.layers1,
            swin_vit.layers2,
            swin_vit.layers3,
            swin_vit.layers4,
        ]

        layer_idx = 0
        for block_list in all_layer_groups:
            for basic_layer in block_list:
                for block in basic_layer.blocks:
                    # --- FFN (mlp) の MoE化 ---
                    mlp = block.mlp
                    if hasattr(mlp, 'fc1') and hasattr(mlp, 'fc2'):
                        embed_dim = mlp.fc1.in_features
                        hidden_dim = mlp.fc1.out_features

                        moe_ffn = LoRAMoEFFN(
                            embed_dim=embed_dim,
                            hidden_dim=hidden_dim,
                            rank=self.config.model.lora_rank,
                            alpha=self.config.model.lora_alpha,
                            pretrained_wi=mlp.fc1.weight.data.clone(),
                            pretrained_wo=mlp.fc2.weight.data.clone(),
                            pretrained_bi=mlp.fc1.bias.data.clone() if mlp.fc1.bias is not None else None,
                            pretrained_bo=mlp.fc2.bias.data.clone() if mlp.fc2.bias is not None else None,
                        )
                        self.moe_ffn_layers.append(moe_ffn)

                        gating = LanguageGuidedGating(
                            embed_dim=embed_dim,
                            clip_embed_dim=self.config.model.clip_embed_dim,
                        )
                        self.gating_modules[f"ffn_{layer_idx}"] = gating

                        # block.mlp をラッパーで置き換え
                        block.mlp = _MoEFFNWrapper(moe_ffn, self)

                    # --- Attention の MoE化 ---
                    attn = block.attn
                    if hasattr(attn, 'qkv') and hasattr(attn, 'proj'):
                        embed_dim = attn.proj.in_features

                        moe_attn = LoRAMoEAttention(
                            embed_dim=embed_dim,
                            num_heads=attn.num_heads if hasattr(attn, 'num_heads') else 8,
                            rank=self.config.model.lora_rank,
                            alpha=self.config.model.lora_alpha,
                            pretrained_qkv_weight=attn.qkv.weight.data.clone(),
                            pretrained_proj_weight=attn.proj.weight.data.clone(),
                            pretrained_qkv_bias=attn.qkv.bias.data.clone() if attn.qkv.bias is not None else None,
                            pretrained_proj_bias=attn.proj.bias.data.clone() if attn.proj.bias is not None else None,
                        )
                        self.moe_attn_layers.append(moe_attn)

                        gating_attn = LanguageGuidedGating(
                            embed_dim=embed_dim,
                            clip_embed_dim=self.config.model.clip_embed_dim,
                        )
                        self.gating_modules[f"attn_{layer_idx}"] = gating_attn

                        # block.attn をラッパーで置き換え
                        block.attn = _MoEAttnWrapper(moe_attn, self)

                    layer_idx += 1

        print(f"[MoE挿入完了] FFN: {len(self.moe_ffn_layers)}層, "
              f"Attention: {len(self.moe_attn_layers)}層")

    def prepare_step(
        self,
        step: int,
        text_description: str,
        num_classes: int,
        device: str = "cuda",
    ):
        """
        新しい学習ステップの準備

        1. 前ステップのエキスパートを凍結
        2. 新しいエキスパートを追加
        3. テキストembeddingを生成
        4. セグメンテーションヘッドを更新

        Args:
            step: ステップ番号（1から開始）
            text_description: データセットのテキスト記述
            num_classes: 累積クラス数（背景含む）
            device: デバイス
        """
        assert step == self.current_step + 1, \
            f"ステップは順番に実行してください。現在: {self.current_step}, 要求: {step}"

        # 前ステップのエキスパートを凍結
        if self.current_step > 0:
            for moe_ffn in self.moe_ffn_layers:
                moe_ffn.freeze_expert(self.current_step - 1)
            for moe_attn in self.moe_attn_layers:
                moe_attn.freeze_expert(self.current_step - 1)

        # 新しいエキスパートを追加
        for moe_ffn in self.moe_ffn_layers:
            moe_ffn.add_expert()
        for moe_attn in self.moe_attn_layers:
            moe_attn.add_expert()

        # テキストembeddingを生成
        text_emb = self.clip_encoder.encode(text_description, device=device)
        self.text_embeddings.append(text_emb)

        # セグメンテーションヘッドを更新
        self._update_seg_head(num_classes, device)

        self.current_step = step
        print(f"[Step {step} 準備完了] エキスパート数: {step}, クラス数: {num_classes}")

    def _update_seg_head(self, num_classes: int, device: str = "cuda"):
        """base_model.out を新しいクラス数のConvに差し替える"""
        old_out = self.base_model.out
        if hasattr(old_out, 'conv') and hasattr(old_out.conv, 'conv'):
            in_channels = old_out.conv.conv.in_channels
        else:
            in_channels = self.config.model.feature_size

        new_out = nn.Conv3d(in_channels=in_channels, out_channels=num_classes, kernel_size=1)
        new_out = new_out.to(device)
        for param in new_out.parameters():
            param.requires_grad = True

        self.base_model.out = new_out

    @property
    def seg_head(self) -> nn.Module:
        return self.base_model.out

    def forward(
        self,
        x: torch.Tensor,
        training_step: int = None,
    ) -> torch.Tensor:
        """
        フォワードパス

        block.mlp / block.attn はすでにMoEラッパーに差し替え済みなので、
        self.base_model(x) をそのまま呼ぶだけでMoEが適用される。

        Args:
            x: [batch, C, H, W, D] 入力
            training_step: 学習時は現在のステップ番号、テスト時はNone

        Returns:
            logits: [batch, num_classes, H, W, D]
        """
        if training_step is not None:
            self._current_expert_idx = training_step - 1
        else:
            self._current_expert_idx = self.current_step - 1

        return self.base_model(x)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """現在のステップで学習可能なパラメータを返す"""
        params = []

        expert_idx = self.current_step - 1
        for moe_ffn in self.moe_ffn_layers:
            if expert_idx < moe_ffn.num_experts:
                params.extend([
                    moe_ffn.experts_A_i[expert_idx],
                    moe_ffn.experts_B_i[expert_idx],
                    moe_ffn.experts_A_o[expert_idx],
                    moe_ffn.experts_B_o[expert_idx],
                ])
        for moe_attn in self.moe_attn_layers:
            if expert_idx < moe_attn.num_experts:
                params.extend([
                    moe_attn.experts_A_qkv[expert_idx],
                    moe_attn.experts_B_qkv[expert_idx],
                    moe_attn.experts_A_proj[expert_idx],
                    moe_attn.experts_B_proj[expert_idx],
                ])

        for gating in self.gating_modules.values():
            params.extend(gating.parameters())

        params.extend(self.seg_head.parameters())

        return params

    def count_trainable_params(self) -> Dict[str, int]:
        """学習可能パラメータ数をカテゴリ別に表示"""
        counts = {
            "lora_experts": 0,
            "gating": 0,
            "seg_head": 0,
            "total_trainable": 0,
            "total_frozen": 0,
        }

        for name, param in self.named_parameters():
            if param.requires_grad:
                if "experts" in name:
                    counts["lora_experts"] += param.numel()
                elif "gating" in name:
                    counts["gating"] += param.numel()
                elif "seg_head" in name or "base_model.out" in name:
                    counts["seg_head"] += param.numel()
                counts["total_trainable"] += param.numel()
            else:
                counts["total_frozen"] += param.numel()

        return counts
