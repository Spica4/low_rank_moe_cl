"""
言語ガイド付きゲーティング（Class-level Gating）
論文 Section 2.2 / Figure 3

学習時:
  テキスト記述 → CLIPテキストエンコーダ → テキストembedding
  テキストembedding × (Linear + Sigmoid)(入力) → ゲーティング重み (GW)
  GW × 入力 → エキスパートへの入力

テスト時:
  各エキスパートのテキストembeddingからGWを計算
  Top-1ハードルーティング: 各トークンで重みの大きいエキスパートを選択
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LanguageGuidedGating(nn.Module):
    """
    言語ガイド付きゲーティングモジュール
    
    CLIPテキストエンコーダで生成したembeddingを使い、
    入力トークンごとにどのエキスパートを使うか決定する。
    
    論文 Figure 3:
    - テキスト記述 → CLIP Text Encoder → Emb [1, clip_dim]
    - 入力 x [n, c] → Linear → Sigmoid → GW [n, 1]
    - Emb と 入力を行列積 → Attention weight
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

        # ゲーティング重み計算用の線形層
        self.gate_linear = nn.Linear(embed_dim, 1, bias=True)

    def compute_gating_weights(
        self,
        x: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        単一エキスパートのゲーティング重みを計算
        
        論文 Figure 3 の Training stage:
        x: [batch, n, c] (n = h×w×d for 3D, n = seq_len)
        text_embedding: [1, clip_dim] or [clip_dim]
        
        return: gating_weights [batch, n, 1]
        """
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)

        # テキストembeddingを特徴空間に射影 [1, embed_dim]
        text_feat = self.text_proj(text_embedding)  # [1, embed_dim]

        # 入力とテキストの類似度（行列積）
        # x: [..., c], text_feat: [1, c] → [..., 1]
        # 省略記号 ... により 3D [B, N, C] / 5D [B, D, H, W, C] 両方に対応
        similarity = torch.einsum('...c,mc->...m', x, text_feat)  # [..., 1]

        # ゲーティング重み（sigmoid）
        gating_weights = torch.sigmoid(similarity)  # [..., 1]

        return gating_weights

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

    def forward_test(
        self,
        x: torch.Tensor,
        text_embeddings: List[torch.Tensor],
    ) -> tuple:
        """
        テスト時のフォワードパス（Top-1ハードルーティング）
        
        論文 Figure 3 の Testing stage:
        全エキスパートのテキストembeddingからゲーティング重みを計算し、
        各トークンで最大のエキスパートを選択
        
        x: [batch, n, c]
        text_embeddings: List of [clip_dim] テンソル（各エキスパート）
        
        return:
            routing_weights: [batch, n, num_experts]
            expert_indices: [batch, n] (Top-1のエキスパートインデックス)
        """
        num_experts = len(text_embeddings)
        gating_weights_list = []

        for text_emb in text_embeddings:
            gw = self.compute_gating_weights(x, text_emb)  # [batch, n, 1]
            gating_weights_list.append(gw)

        # [batch, n, num_experts]
        routing_weights = torch.cat(gating_weights_list, dim=-1)

        # Top-1 ハードルーティング
        expert_indices = routing_weights.argmax(dim=-1)  # [batch, n]

        return routing_weights, expert_indices


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
