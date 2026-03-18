"""
設定ファイル: 論文のハイパーパラメータに基づく
Class-level Continual Learning の設定
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """モデル構成の設定"""
    # Swin-UNETR 関連
    img_size: tuple = (96, 96, 96)       # 3D入力サイズ
    in_channels: int = 1                   # 入力チャンネル数（CT）
    feature_size: int = 48                 # Swin-UNETRの特徴サイズ

    # LoRA 関連
    lora_rank: int = 8                     # 低ランクの次元 r（論文: r=8）
    lora_alpha: float = 1.0                # LoRAのスケーリング係数
    lora_dropout: float = 0.0              # LoRAドロップアウト

    # MoE 関連
    num_experts: int = 2                   # エキスパート数（Step数に対応）
    top_k: int = 1                         # Top-K ルーティング（論文: Top-1）

    # CLIP テキストエンコーダ
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_embed_dim: int = 512              # CLIPテキストembedding次元

    # 事前学習済み重みパス（Swin-UNETR SSL pretrained）
    # ダウンロード: https://github.com/Project-MONAI/MONAI-extra-test-data/releases
    # 例: model_swinvit.pt
    pretrained_weights_path: str = "/deeparea/sokabe/weight/model_swinvit.pt"     # 空文字列の場合はスクラッチ学習


@dataclass
class TrainConfig:
    """学習設定"""
    # Step1 (例: BTCV)
    step1_epochs: int = 500
    step1_lr: float = 1e-3
    step1_weight_decay: float = 1e-5
    step1_batch_size: int = 1

    # Step2 (例: LiTS)
    step2_epochs: int = 200
    step2_lr: float = 1e-3
    step2_weight_decay: float = 1e-5
    step2_batch_size: int = 1

    # 共通
    warmup_epochs: int = 10
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    # チェックポイント
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 50


@dataclass
class DataConfig:
    """データセット設定（差し替え可能）"""
    # Step1 データセット
    step1_name: str = "BTCV"
    step1_train_dir: str = "/deeparea/sokabe/Dataset/BTCV/Abdomen/train"
    step1_val_dir: str = "/deeparea/sokabe/Dataset/BTCV/Abdomen/validation"
    step1_num_classes: int = 14             # 例: BTCV = 13クラス + 背景

    # Step2 データセット
    step2_name: str = "LiTS"
    step2_train_dir: str = "/deeparea/sokabe/Dataset/LiTS/train"
    step2_val_dir: str = "/deeparea/sokabe/Dataset/LiTS/validation"
    step2_num_classes: int = 2              # 例: LiTS = 肝腫瘍 + 背景（新規クラス）

    # Step1 + Step2 の合計クラス数
    total_num_classes: int = 15             # Step1 + Step2の新規クラス

    # テキスト記述（CLIPゲーティング用）
    # ユーザーが自分のデータセットに合わせて書き換える
    step1_text_description: str = (
        "Medical imaging dataset for abdominal organs with the following "
        "label definitions: 0.background; 1.spleen; 2.right kidney; "
        "3.left kidney; 4.gallbladder; 5.esophagus; 6.liver; 7.stomach; "
        "8.aorta; 9.inferior vena cava; 10.portal vein and splenic vein; "
        "11.pancreas; 12.right adrenal gland; 13.left adrenal gland."
    )
    step2_text_description: str = (
        "Medical imaging dataset for liver tumor segmentation with the "
        "following label definitions: 0.background; 1.liver tumor."
    )

    # 前処理
    spatial_size: tuple = (96, 96, 96)
    num_workers: int = 4


@dataclass
class Config:
    """全体設定"""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True                   # Mixed Precision Training
