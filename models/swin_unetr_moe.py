"""
Swin-UNETR + Low-Rank MoE 統合モデル
論文: 3D Swin Transformerを拡張し、MoE構造を組み込む

MONAIのSwinUNETRをベースに:
1. Swin Transformer Blockの FFN層 と Attention層 にLoRA MoEを挿入
2. 言語ガイド付きゲーティングでエキスパートを選択
3. 各ステップでエキスパートを追加・凍結
"""
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from copy import deepcopy

from .lora_moe import LoRAMoEFFN, LoRAMoEAttention
from .language_gating import LanguageGuidedGating, CLIPTextEncoder


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

        # CLIPテキストエンコーダ
        self.clip_encoder = CLIPTextEncoder(config.model.clip_model_name)

        # ベースモデル: MONAI SwinUNETR
        self._build_base_model()

        # MoEモジュールを格納
        self.moe_ffn_layers: nn.ModuleList = nn.ModuleList()
        self.moe_attn_layers: nn.ModuleList = nn.ModuleList()

        # 言語ガイドゲーティング
        self.gating_modules: nn.ModuleDict = nn.ModuleDict()

        # MoE挿入位置を特定してモジュールを作成
        self._inject_moe_layers()

    def _build_base_model(self):
        """MONAIのSwinUNETRを構築"""
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError:
            raise ImportError(
                "MONAIが必要です: pip install monai"
            )

        self.base_model = SwinUNETR(
            img_size=self.config.model.img_size,
            in_channels=self.config.model.in_channels,
            out_channels=self.config.data.step1_num_classes,  # 初期クラス数
            feature_size=self.config.model.feature_size,
            use_checkpoint=False,
        )

        # ベースモデルのパラメータを凍結
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _inject_moe_layers(self):
        """
        Swin Transformer BlockのFFNとAttentionにMoEを挿入

        MONAIのSwinUNETRの構造:
        swinViT.layers1[j].mlp  → FFN
        swinViT.layers1[j].attn → Attention
        (layers1〜layers4 がステージごとに存在)
        """
        swin_vit = self.base_model.swinViT

        all_layer_groups = [
            swin_vit.layers1,
            swin_vit.layers2,
            swin_vit.layers3,
            swin_vit.layers4,
        ]

        layer_idx = 0
        for i, blocks in enumerate(all_layer_groups):
            for j, block in enumerate(blocks):
                # FFN (mlp) の MoE化
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

                    # ゲーティングモジュール
                    gating = LanguageGuidedGating(
                        embed_dim=embed_dim,
                        clip_embed_dim=self.config.model.clip_embed_dim,
                    )
                    self.gating_modules[f"ffn_{layer_idx}"] = gating

                # Attention の MoE化
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

        # セグメンテーションヘッドを更新（クラス数が変わる場合）
        self._update_seg_head(num_classes)

        self.current_step = step
        print(f"[Step {step} 準備完了] エキスパート数: {step}, クラス数: {num_classes}")

    def _update_seg_head(self, num_classes: int):
        """セグメンテーションヘッドのクラス数を更新"""
        # SwinUNETRの最終出力層を差し替え
        old_out = self.base_model.out
        if hasattr(old_out, 'conv') and hasattr(old_out.conv, 'conv'):
            in_channels = old_out.conv.conv.in_channels
        else:
            # MONAI版に応じて適宜調整
            in_channels = self.config.model.feature_size

        self.seg_head = nn.Conv3d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1,
        )
        # セグメンテーションヘッドは学習可能
        for param in self.seg_head.parameters():
            param.requires_grad = True

    def _forward_with_moe(
        self,
        x: torch.Tensor,
        expert_idx: int = None,
        use_routing: bool = False,
    ) -> torch.Tensor:
        """
        MoEを組み込んだフォワードパス
        
        Swin-UNETRのエンコーダ部分を通しながら、
        各ブロックでMoE FFN/Attentionを適用
        
        Args:
            x: [batch, C, H, W, D] 入力ボリューム
            expert_idx: 学習時に使用するエキスパートインデックス
            use_routing: テスト時のルーティングを使うかどうか
        """
        swin_vit = self.base_model.swinViT

        # SwinViTのパッチ埋め込み
        if hasattr(swin_vit, 'patch_embed'):
            x = swin_vit.patch_embed(x)
        elif hasattr(swin_vit, 'patch_embedding'):
            x = swin_vit.patch_embedding(x)

        # 各Swin Transformer Layerを通過
        hidden_states = []
        layer_idx = 0

        all_layer_groups = [
            swin_vit.layers1,
            swin_vit.layers2,
            swin_vit.layers3,
            swin_vit.layers4,
        ]

        for i, blocks in enumerate(all_layer_groups):
            for j, block in enumerate(blocks):
                # --- Attention with MoE ---
                if layer_idx < len(self.moe_attn_layers):
                    moe_attn = self.moe_attn_layers[layer_idx]

                    if use_routing and len(self.text_embeddings) > 1:
                        # テスト時: ルーティング
                        gating = self.gating_modules[f"attn_{layer_idx}"]
                        routing_weights, expert_indices = gating.forward_test(
                            x, self.text_embeddings
                        )
                        # 各トークンをTop-1エキスパートで処理
                        attn_out = self._route_attention(
                            moe_attn, x, expert_indices
                        )
                    else:
                        # 学習時: 指定エキスパートのみ
                        idx = expert_idx if expert_idx is not None else self.current_step - 1
                        if moe_attn.num_experts > 0:
                            attn_out = moe_attn.forward_single_expert(x, idx)
                        else:
                            attn_out = x

                    # Residual connection (Swin Transformer style)
                    x = x + attn_out

                # --- FFN with MoE ---
                if layer_idx < len(self.moe_ffn_layers):
                    moe_ffn = self.moe_ffn_layers[layer_idx]

                    if use_routing and len(self.text_embeddings) > 1:
                        gating = self.gating_modules[f"ffn_{layer_idx}"]
                        routing_weights, expert_indices = gating.forward_test(
                            x, self.text_embeddings
                        )
                        ffn_out = moe_ffn.forward_routed(x, routing_weights)
                    else:
                        idx = expert_idx if expert_idx is not None else self.current_step - 1
                        if moe_ffn.num_experts > 0:
                            ffn_out = moe_ffn.forward_single_expert(x, idx)
                        else:
                            ffn_out = x

                    x = x + ffn_out

                layer_idx += 1

            hidden_states.append(x)

            # Patch Merging (downsample): MONAIはdownsample1〜3を持つ
            downsample = getattr(swin_vit, f"downsample{i + 1}", None)
            if downsample is not None:
                x = downsample(x)

        return hidden_states

    def _route_attention(
        self,
        moe_attn: LoRAMoEAttention,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Top-1ルーティングでAttention出力を計算"""
        output = torch.zeros_like(x)
        for expert_idx in range(moe_attn.num_experts):
            mask = (expert_indices == expert_idx)
            if not mask.any():
                continue
            expert_out = moe_attn.forward_single_expert(x, expert_idx)
            mask_expanded = mask.unsqueeze(-1).expand_as(output)
            output = output + expert_out * mask_expanded.float()
        return output

    def forward(
        self,
        x: torch.Tensor,
        training_step: int = None,
    ) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: [batch, C, H, W, D] 入力
            training_step: 学習時は現在のステップ番号、テスト時はNone
            
        Returns:
            logits: [batch, num_classes, H, W, D]
        """
        use_routing = (training_step is None)

        if training_step is not None:
            expert_idx = training_step - 1
        else:
            expert_idx = None

        # SwinViTエンコーダ（MoE付き）
        hidden_states = self._forward_with_moe(
            x, expert_idx=expert_idx, use_routing=use_routing
        )

        # デコーダ（UNETRデコーダ部分）
        # 注: ここはSwinUNETRのデコーダをそのまま使用
        # 実際の実装ではMONAIのデコーダ構造に合わせて調整が必要
        logits = self._decode(hidden_states, x)

        return logits

    def _decode(
        self,
        hidden_states: List[torch.Tensor],
        original_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        UNETRスタイルのデコーダ
        
        hidden_statesをスキップ接続としてデコーダに渡す。
        実際の実装ではMONAIのSwinUNETRデコーダ構造に合わせる。
        """
        # 簡易実装: 最終隠れ状態を使用
        # 本番ではbase_modelのデコーダ部分を活用
        dec = self.base_model
        
        # MONAIのSwinUNETRのデコーダ呼び出し
        # (実装はMONAIのバージョンに依存するため、適宜調整)
        if len(hidden_states) >= 4:
            enc0 = hidden_states[0]
            enc1 = hidden_states[1]
            enc2 = hidden_states[2]
            enc3 = hidden_states[3]

            # デコーダパス（SwinUNETRのデコーダ構造に従う）
            # ここは概略的な実装 - 実際のMONAI APIに合わせて調整
            try:
                dec0 = dec.decoder5(enc3, None)  
                dec1 = dec.decoder4(dec0, enc2)
                dec2 = dec.decoder3(dec1, enc1)
                dec3 = dec.decoder2(dec2, enc0)
                logits = self.seg_head(dec3)
            except (AttributeError, TypeError):
                # フォールバック: ベースモデルの出力層を使用
                final_feat = hidden_states[-1]
                # reshape and upsample
                B = original_input.shape[0]
                logits = self.seg_head(
                    final_feat.permute(0, 2, 1).reshape(
                        B, -1, *[s // 32 for s in self.config.model.img_size]
                    )
                )
                logits = nn.functional.interpolate(
                    logits, size=original_input.shape[2:], mode='trilinear'
                )
        else:
            raise ValueError(f"hidden_statesが不足: {len(hidden_states)}")

        return logits

    def get_trainable_params(self) -> List[nn.Parameter]:
        """現在のステップで学習可能なパラメータを返す"""
        params = []

        # 現在のエキスパートのLoRAパラメータ
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

        # ゲーティングモジュール
        for gating in self.gating_modules.values():
            params.extend(gating.parameters())

        # セグメンテーションヘッド
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
                elif "seg_head" in name:
                    counts["seg_head"] += param.numel()
                counts["total_trainable"] += param.numel()
            else:
                counts["total_frozen"] += param.numel()

        return counts
