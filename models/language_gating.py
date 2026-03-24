"""
言語ガイド付きゲーティング（Class-level Gating）
論文 Section 2.2 / Figure 3

学習時:
  テキスト記述 → CLIPテキストエンコーダ → テキストembedding (固定)
  GW = sigmoid(x · linear(text_embedding))
  GW × 入力 → エキスパートへの入力

テスト時:
  各エキスパートのGWを計算してTop-1ハードルーティング:
  argmax([GW_step0, GW_step1])

[初期化時の挙動]
  text_proj ≈ 0 (ランダム初期化) → sigmoid(x·0) = 0.5
  gated_x = 0.5 * x → エキスパートに十分な信号が伝わり学習が進む

[注意: 別実装との比較]
  入力を先にCLIP空間に射影する実装 GW = text_emb · sigmoid(input_proj(x)) は
  L2正規化済みtext_embの要素和 ≈ 0 のため GW ≈ 0 になり学習不能になる。
  本実装 (text_projを先に適用) が学習安定性の観点で正しい。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class LanguageGuidedGating(nn.Module):
    """
    言語ガイド付きゲーティングモジュール

    論文 Figure 3:
    - テキスト記述 → CLIP Text Encoder → text_emb [clip_dim]  (固定)
    - text_emb → Linear (text_proj) → text_feat [embed_dim]
    - GW = sigmoid(einsum(x, text_feat)) → [..., 1]  in (0, 1)
    - gated_x = GW * x → エキスパートへ

    未学習時: text_feat ≈ 0 → sigmoid(0) = 0.5 → gated_x = 0.5 * x
    学習後: text_feat がドメイン方向を向き、ドメイン特徴でGW > 0.5 になる
    """

    def __init__(
        self,
        embed_dim: int,
        clip_embed_dim: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.clip_embed_dim = clip_embed_dim

        # テキストembeddingを特徴次元に射影
        self.text_proj = nn.Linear(clip_embed_dim, embed_dim, bias=False)

    def compute_gating_weights(
        self,
        x: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        ゲーティング重みを計算

        x: [..., embed_dim]
        text_embedding: [clip_dim] or [1, clip_dim]
        return: GW [..., 1]  in (0, 1)
        """
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)

        text_feat = self.text_proj(text_embedding)               # [1, embed_dim]
        logit = torch.einsum('...c,mc->...m', x, text_feat)      # [..., 1]
        return torch.sigmoid(logit)

    def forward_train(
        self,
        x: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        学習時のフォワードパス: GW * x を返す

        x: [..., embed_dim]
        text_embedding: [clip_dim]
        return: gated_x [..., embed_dim]
        """
        gw = self.compute_gating_weights(x, text_embedding)
        return x * gw


class CLIPTextEncoder(nn.Module):
    """
    CLIPテキストエンコーダのラッパー

    データセットのテキスト記述からembeddingを生成する。

    使用例:
        encoder = CLIPTextEncoder()
        emb = encoder.encode("BTCV dataset contains...")
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """遅延ロード（初回使用時にモデルをロード）"""
        if self._model is None:
            try:
                from transformers import CLIPModel, CLIPTokenizer
                self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
                self._model = CLIPModel.from_pretrained(self.model_name)
                # テキストエンコーダは凍結
                for param in self._model.parameters():
                    param.requires_grad = False
                self._model.eval()
            except ImportError:
                raise ImportError(
                    "transformersライブラリが必要です: pip install transformers"
                )

    @torch.no_grad()
    def encode(self, text: str, device: str = "cuda") -> torch.Tensor:
        """
        テキストをCLIP embeddingに変換

        text: データセットの説明テキスト
        return: [clip_embed_dim] テンソル
        """
        self._load_model()
        self._model = self._model.to(device)

        inputs = self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self._model.get_text_features(**inputs)
        # 正規化
        text_embedding = outputs / outputs.norm(dim=-1, keepdim=True)
        return text_embedding.squeeze(0)  # [clip_embed_dim]

    @torch.no_grad()
    def encode_batch(self, texts: List[str], device: str = "cuda") -> List[torch.Tensor]:
        """複数テキストを一括エンコード"""
        return [self.encode(text, device) for text in texts]
