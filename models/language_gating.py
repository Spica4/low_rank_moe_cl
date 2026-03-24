"""
言語ガイド付きゲーティング（Class-level Gating）
論文 Section 2.2 / Figure 3

学習時:
  テキスト記述 → CLIPテキストエンコーダ → テキストembedding (固定)
  入力 x → Linear → Sigmoid → x_proj  [論文の順序: 入力を先に変換]
  GW = text_embedding · x_proj         [内積: 入力がどれだけテキストに類似するか]
  GW × 入力 → エキスパートへの入力

テスト時:
  各エキスパートのGWを計算してTop-1ハードルーティング

[設計上の注意]
  入力を先にCLIP空間に射影してからテキストembeddingと内積を取る。
  テキストembeddingを特徴空間に射影する逆順は NG。
  理由: 未学習時にGW≈0 (中立) になるため、学習後に正しくドメイン識別できる。
  逆順だとGW≈0.5 (バイアス)になり、Step1未学習のGW_step0が常に不利になる。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LanguageGuidedGating(nn.Module):
    """
    言語ガイド付きゲーティングモジュール

    論文 Figure 3:
    - テキスト記述 → CLIP Text Encoder → text_emb [clip_dim]  (固定)
    - 入力 x [..., embed_dim] → Linear → Sigmoid → x_proj [..., clip_dim]
    - GW = einsum(x_proj, text_emb) → [..., 1]  (外側のSigmoidなし)
    """

    def __init__(
        self,
        embed_dim: int,
        clip_embed_dim: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.clip_embed_dim = clip_embed_dim

        # 入力特徴量をCLIP埋め込み空間に射影
        # 論文: "Linear + Sigmoid を入力に適用してからテキストembeddingと行列積"
        self.input_proj = nn.Linear(embed_dim, clip_embed_dim, bias=True)

    def compute_gating_weights(
        self,
        x: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        単一エキスパートのゲーティング重みを計算

        論文 Figure 3 の Training stage:
          x: [..., embed_dim]
          text_embedding: [clip_dim] or [1, clip_dim]

        return: GW [..., 1]
          未学習時: input_proj ≈ 0 → sigmoid(0) = 0.5 → GW = text_emb · 0.5
          L2正規化済みtext_embの要素平均 ≈ 0 → GW ≈ 0 (中立)
          学習後: ドメイン特徴に対して正、異ドメインに対して小/負の値
        """
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)  # [1, clip_dim]

        # 入力をCLIP空間に射影 → Sigmoid [..., clip_dim]
        x_proj = torch.sigmoid(self.input_proj(x))  # [..., clip_dim]

        # テキストembeddingとの内積 → GW [..., 1]
        gating_weights = torch.einsum('...c,mc->...m', x_proj, text_embedding)  # [..., 1]

        return gating_weights

    def compute_logits(
        self,
        x: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        テスト時のルーティング用にGWをそのまま返す（compute_gating_weightsと同一）

        x: [..., embed_dim]
        text_embedding: [clip_dim] or [1, clip_dim]
        return: GW [..., 1]
        """
        return self.compute_gating_weights(x, text_embedding)

    def forward_train(
        self,
        x: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        学習時のフォワードパス
        
        テキストembeddingに基づくゲーティング重みで入力を変調
        
        x: [batch, n, c]
        text_embedding: [clip_dim]
        return: gated_x [batch, n, c]
        """
        gw = self.compute_gating_weights(x, text_embedding)  # [batch, n, 1]
        return x * gw  # [batch, n, c]


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
