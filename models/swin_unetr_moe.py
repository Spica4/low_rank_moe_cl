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

    学習時: 固定エキスパート（_current_expert_idx）を使用
    テスト時: LanguageGuidedGating による CLIP Top-1 ルーティングを使用
    """

    def __init__(
        self,
        moe_ffn: LoRAMoEFFN,
        model_ref: "SwinUNETRMoE",
        gating_key: str,
    ):
        super().__init__()
        self.moe_ffn = moe_ffn
        self._model_ref = weakref.ref(model_ref)
        self._gating_key = gating_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.moe_ffn.num_experts == 0:
            h = self.moe_ffn.activation(self.moe_ffn.base_wi(x))
            return self.moe_ffn.base_wo(h)

        model = self._model_ref()
        gating = model.gating_modules[self._gating_key]

        if model._is_test_mode and self.moe_ffn.num_experts > 1:
            # テスト時: CLIP Top-1 ルーティング（per-token）
            routing_weights, _ = gating.forward_test(x, model.text_embeddings)
            return self.moe_ffn.forward_routed(x, routing_weights)

        # 学習時: 現ステップのテキスト embedding でゲーティングを適用してから固定エキスパートへ
        expert_idx = model._current_expert_idx
        text_emb = model.text_embeddings[expert_idx]
        gated_x = gating.forward_train(x, text_emb)
        return self.moe_ffn.forward_single_expert(gated_x, expert_idx)


class _MoEAttnWrapper(nn.Module):
    """
    SwinTransformerBlock.attn をMoE Attentionに差し替えるラッパー。

    学習時: 固定エキスパート（_current_expert_idx）を使用
    テスト時: LanguageGuidedGating による CLIP Top-1 ルーティングを使用
    """

    def __init__(
        self,
        moe_attn: LoRAMoEAttention,
        model_ref: "SwinUNETRMoE",
        gating_key: str,
    ):
        super().__init__()
        self.moe_attn = moe_attn
        self._model_ref = weakref.ref(model_ref)
        self._gating_key = gating_key

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        if self.moe_attn.num_experts == 0:
            # エキスパート未追加時はベース重みのみで計算
            B, N, C = x.shape
            qkv = self.moe_attn.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.moe_attn.num_heads, self.moe_attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = self.moe_attn.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            if self.moe_attn.relative_position_bias_table is not None:
                relative_position_bias = self.moe_attn.relative_position_bias_table[
                    self.moe_attn.relative_position_index[:N, :N].reshape(-1)
                ].reshape(N, N, -1).permute(2, 0, 1).contiguous()
                attn = attn + relative_position_bias.unsqueeze(0)
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B // nW, nW, self.moe_attn.num_heads, N, N)
                attn = attn + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.moe_attn.num_heads, N, N)
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.moe_attn.proj(out)

        model = self._model_ref()
        gating = model.gating_modules[self._gating_key]

        if model._is_test_mode and self.moe_attn.num_experts > 1:
            # テスト時: CLIP Top-1 ルーティング（per-token）
            routing_weights, _ = gating.forward_test(x, model.text_embeddings)
            return self.moe_attn.forward_routed(x, routing_weights, attn_mask=mask)

        # 学習時: 現ステップのテキスト embedding でゲーティングを適用してから固定エキスパートへ
        expert_idx = model._current_expert_idx
        text_emb = model.text_embeddings[expert_idx]
        gated_x = gating.forward_train(x, text_emb)
        return self.moe_attn.forward_single_expert(gated_x, expert_idx, mask=mask)


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

        # forward時に各ラッパーが参照するエキスパートインデックス（学習時）
        self._current_expert_idx: int = 0
        # テスト時は True → ラッパーが CLIP ルーティングを使用する
        self._is_test_mode: bool = False

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

        # 事前学習済み重みのロード
        pretrained_path = self.config.model.pretrained_weights_path
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        else:
            print("[警告] 事前学習済み重みが指定されていません。スクラッチ学習を行います。")
            print("  config.model.pretrained_weights_path にパスを設定することを推奨します。")

        # ベースモデルのパラメータを凍結
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self, pretrained_path: str):
        """
        事前学習済み重みをロードする。
        以下の2つの形式を自動判別する:

        1. フルSwinUNETRチェックポイント:
           encoder/decoderブロックを含む完全な重み。
           セグメンテーション学習済みモデルが該当。
           → load_state_dict() で全層をロード

        2. SwinViTのみのチェックポイント:
           SSL事前学習済みのSwinViTエンコーダ重みのみ。
           model_swinvit.pt が該当。
           → MONAI の load_from() でエンコーダのみロード
           ※ encoder/decoderブロックはランダム初期化のまま
        """
        raw = torch.load(pretrained_path, map_location="cpu")

        # state_dict を取り出す（チェックポイント形式の差異を吸収）
        if isinstance(raw, dict):
            if "state_dict" in raw:
                state_dict = raw["state_dict"]
            elif "model" in raw:
                state_dict = raw["model"]
            else:
                state_dict = raw
        else:
            state_dict = raw

        # "module." プレフィックスを除去（DataParallel 対応）
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }

        # フルSwinUNETRか SwinViTのみか を判定
        has_encoder_blocks = any(
            k.startswith(("encoder1", "encoder2", "encoder3", "encoder4", "encoder10"))
            for k in state_dict
        )
        has_decoder_blocks = any(
            k.startswith(("decoder2", "decoder3", "decoder4", "decoder5"))
            for k in state_dict
        )

        if has_encoder_blocks or has_decoder_blocks:
            self._load_full_swinunetr(state_dict, pretrained_path)
        else:
            self._load_swinvit_only(raw, pretrained_path)

    def _load_full_swinunetr(self, state_dict: dict, path: str):
        """フルSwinUNETRチェックポイントをロード"""
        # out層は後で _update_seg_head() で差し替えるため除外
        filtered = {
            k: v for k, v in state_dict.items() if not k.startswith("out")
        }

        missing, unexpected = self.base_model.load_state_dict(
            filtered, strict=False
        )

        # ロード結果を表示
        loaded_parts = []
        if any(k.startswith("swinViT") for k in state_dict):
            loaded_parts.append("SwinViT")
        if any(k.startswith(("encoder1", "encoder2", "encoder3", "encoder4")) for k in state_dict):
            loaded_parts.append("encoder blocks")
        if any(k.startswith("encoder10") for k in state_dict):
            loaded_parts.append("bottleneck")
        if any(k.startswith(("decoder2", "decoder3", "decoder4", "decoder5")) for k in state_dict):
            loaded_parts.append("decoder blocks")

        print(f"[事前学習済み重みをロード（フルSwinUNETR）] {path}")
        print(f"  ロード済み: {', '.join(loaded_parts)}")
        if missing:
            # out層の除外分を差し引く
            non_out_missing = [k for k in missing if not k.startswith("out")]
            if non_out_missing:
                print(f"  未ロード: {len(non_out_missing)}個のパラメータ")
        if unexpected:
            print(f"  無視: {len(unexpected)}個の不明なキー")

    def _load_swinvit_only(self, raw_weights: dict, path: str):
        """SwinViTのみのチェックポイントをロード（MONAI load_from 経由）"""
        if isinstance(raw_weights, dict) and "state_dict" not in raw_weights:
            raw_weights = {"state_dict": raw_weights}
        self.base_model.load_from(weights=raw_weights)
        print(f"[事前学習済み重みをロード（SwinViTのみ）] {path}")
        print(f"  注意: encoder/decoderブロックの事前学習済み重みは含まれていません。")
        print(f"  → encoder/decoderはランダム初期化のまま凍結されます。")
        print(f"  → フルSwinUNETRチェックポイントの使用を推奨します。")

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
                    # MONAI MLPBlock は linear1/linear2 を使用 (fc1/fc2 ではない)
                    _fc1 = getattr(mlp, 'linear1', None) or getattr(mlp, 'fc1', None)
                    _fc2 = getattr(mlp, 'linear2', None) or getattr(mlp, 'fc2', None)
                    if _fc1 is not None and _fc2 is not None:
                        embed_dim = _fc1.in_features
                        hidden_dim = _fc1.out_features

                        moe_ffn = LoRAMoEFFN(
                            embed_dim=embed_dim,
                            hidden_dim=hidden_dim,
                            rank=self.config.model.lora_rank,
                            alpha=self.config.model.lora_alpha,
                            pretrained_wi=_fc1.weight.data.clone(),
                            pretrained_wo=_fc2.weight.data.clone(),
                            pretrained_bi=_fc1.bias.data.clone() if _fc1.bias is not None else None,
                            pretrained_bo=_fc2.bias.data.clone() if _fc2.bias is not None else None,
                        )
                        self.moe_ffn_layers.append(moe_ffn)

                        gating = LanguageGuidedGating(
                            embed_dim=embed_dim,
                            clip_embed_dim=self.config.model.clip_embed_dim,
                        )
                        self.gating_modules[f"ffn_{layer_idx}"] = gating

                        # block.mlp をラッパーで置き換え
                        block.mlp = _MoEFFNWrapper(moe_ffn, self, f"ffn_{layer_idx}")

                    # --- Attention の MoE化 ---
                    attn = block.attn
                    if hasattr(attn, 'qkv') and hasattr(attn, 'proj'):
                        embed_dim = attn.proj.in_features

                        # 相対位置バイアスをコピー（Swin Transformer の必須コンポーネント）
                        rel_pos_bias_table = (
                            attn.relative_position_bias_table.data.clone()
                            if hasattr(attn, 'relative_position_bias_table')
                            else None
                        )
                        rel_pos_index = (
                            attn.relative_position_index.clone()
                            if hasattr(attn, 'relative_position_index')
                            else None
                        )

                        moe_attn = LoRAMoEAttention(
                            embed_dim=embed_dim,
                            num_heads=attn.num_heads if hasattr(attn, 'num_heads') else 8,
                            rank=self.config.model.lora_rank,
                            alpha=self.config.model.lora_alpha,
                            pretrained_qkv_weight=attn.qkv.weight.data.clone(),
                            pretrained_proj_weight=attn.proj.weight.data.clone(),
                            pretrained_qkv_bias=attn.qkv.bias.data.clone() if attn.qkv.bias is not None else None,
                            pretrained_proj_bias=attn.proj.bias.data.clone() if attn.proj.bias is not None else None,
                            relative_position_bias_table=rel_pos_bias_table,
                            relative_position_index=rel_pos_index,
                        )
                        self.moe_attn_layers.append(moe_attn)

                        gating_attn = LanguageGuidedGating(
                            embed_dim=embed_dim,
                            clip_embed_dim=self.config.model.clip_embed_dim,
                        )
                        self.gating_modules[f"attn_{layer_idx}"] = gating_attn

                        # block.attn をラッパーで置き換え
                        block.attn = _MoEAttnWrapper(moe_attn, self, f"attn_{layer_idx}")

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
        """
        base_model.out を新しいクラス数の Conv3d に差し替える。
        旧ヘッドが持つクラス分の重みは新ヘッドへコピーし、
        新規クラス分のみランダム初期化とする。
        """
        old_out = self.base_model.out

        # 旧ヘッドの Conv3d を取得（初回は MONAI UnetOutBlock、以降は plain Conv3d）
        if isinstance(old_out, nn.Conv3d):
            old_conv = old_out
        elif hasattr(old_out, 'conv') and hasattr(old_out.conv, 'conv'):
            old_conv = old_out.conv.conv
        else:
            old_conv = None

        if old_conv is not None:
            in_channels = old_conv.in_channels
            old_num_classes = old_conv.out_channels
            old_weight = old_conv.weight.data.clone()
            old_bias = old_conv.bias.data.clone() if old_conv.bias is not None else None
        else:
            in_channels = self.config.model.feature_size
            old_num_classes = 0
            old_weight = None
            old_bias = None

        new_out = nn.Conv3d(in_channels=in_channels, out_channels=num_classes, kernel_size=1)
        new_out = new_out.to(device)

        # 旧クラスの学習済み重みを引き継ぐ
        if old_weight is not None and 0 < old_num_classes <= num_classes:
            with torch.no_grad():
                new_out.weight.data[:old_num_classes] = old_weight.to(device)
                if old_bias is not None and new_out.bias is not None:
                    new_out.bias.data[:old_num_classes] = old_bias.to(device)
            print(f"  [セグヘッド更新] {old_num_classes}クラスの重みを引き継ぎ、"
                  f"{num_classes - old_num_classes}クラスをランダム初期化")
        else:
            print(f"  [セグヘッド更新] {num_classes}クラスをランダム初期化")

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
            # 学習時: 指定ステップのエキスパートを固定使用
            self._current_expert_idx = training_step - 1
            self._is_test_mode = False
        else:
            # テスト時: CLIP ゲーティングによるルーティングを使用
            self._is_test_mode = True

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
            "frozen_swinvit": 0,
            "frozen_encoder_blocks": 0,
            "frozen_decoder_blocks": 0,
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
                if "swinViT" in name:
                    counts["frozen_swinvit"] += param.numel()
                elif "base_model.encoder" in name:
                    counts["frozen_encoder_blocks"] += param.numel()
                elif "base_model.decoder" in name:
                    counts["frozen_decoder_blocks"] += param.numel()
                counts["total_frozen"] += param.numel()

        return counts
