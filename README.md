# Low-Rank Mixture-of-Experts for Continual Medical Image Segmentation
## Class-level Continual Learning 再現実装

### プロジェクト構成
```
low_rank_moe_cl/
├── README.md
├── config.py              # ハイパーパラメータ・設定
├── models/
│   ├── __init__.py
│   ├── lora_layers.py     # LoRA線形層・LoRA FFN
│   ├── lora_moe.py        # Low-Rank MoE モジュール（FFN + Attention）
│   ├── language_gating.py # 言語ガイド付きゲーティング（CLIP）
│   └── swin_unetr_moe.py  # Swin-UNETR + MoE 統合モデル
├── data/
│   ├── __init__.py
│   └── dataset.py         # データローダー（差し替え可能）
├── train.py               # 学習ループ（Step1 / Step2）
├── test.py                # テストループ
└── utils.py               # ユーティリティ（メトリクス等）
```

### 手法の概要（Class-level CL）
1. **Step1**: BTCV等のデータセットでExpert1を学習（ベースモデルは凍結）
2. **Step2**: Expert1を凍結し、LiTS等の新データでExpert2を学習
3. **テスト時**: CLIPテキストembeddingによるTop-1ハードルーティングで
   各トークンを適切なエキスパートに振り分け

### 主要コンポーネント
- **LoRA層**: W0 + BA の低ランク分解（rank=8）
- **MoE FFN**: 各エキスパートがLoRA FFNで構成
- **言語ガイドゲーティング**: CLIPテキストエンコーダ → Attention → Top-1ルーティング
- **ベースモデル**: 3D Swin-UNETR（MONAI）
