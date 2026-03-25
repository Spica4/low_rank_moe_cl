"""
学習スクリプト: Class-level Continual Learning

論文の学習パイプライン:
  Step1: BTCV等のデータセットでExpert1を学習
  Step2: Expert1を凍結し、LiTS等の新データでExpert2を学習

使い方:
  python train.py --step 1
  python train.py --step 2 --resume checkpoints/step1_best.pth
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from monai.inferers import sliding_window_inference

from config import Config, ModelConfig, TrainConfig, DataConfig
from models.swin_unetr_moe import SwinUNETRMoE
from data.dataset import get_dataloader
from utils import (
    dice_score, DiceCELoss,
    save_checkpoint, load_checkpoint,
    TrainingLogger,
)


def set_seed(seed: int):
    """再現性のためシードを固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int = 10):
    """
    コサイン学習率スケジューラ + 線形ウォームアップ
    論文: cosine learning rate scheduler with 10-epoch linear warm-up
    """
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    return scheduler


def _get_amp_dtype(amp_dtype: str):
    """設定文字列を torch dtype に変換"""
    if amp_dtype == "bf16":
        return torch.bfloat16
    elif amp_dtype == "fp16":
        return torch.float16
    else:
        return torch.float32


def train_one_epoch(
    model: SwinUNETRMoE,
    dataloader,
    optimizer,
    criterion,
    scaler,
    device: str,
    step: int,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
):
    """1エポックの学習"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    dtype = _get_amp_dtype(amp_dtype)

    for batch_data in dataloader:
        # RandCropByPosNegLabeld(num_samples>1) は list[dict] を返す場合がある
        if isinstance(batch_data, list):
            batch_data = {
                k: torch.cat([d[k] for d in batch_data], dim=0)
                for k in batch_data[0].keys()
            }
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type="cuda", dtype=dtype):
                logits = model(images, training_step=step)
                loss = criterion(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
                optimizer.step()
        else:
            logits = model(images, training_step=step)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: SwinUNETRMoE,
    dataloader,
    num_classes: int,
    device: str,
):
    """検証"""
    model.eval()
    all_dice = {}

    for batch_data in dataloader:
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        logits = sliding_window_inference(
            inputs=images,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=lambda x: model(x, training_step=None),
            overlap=0.5,
        )

        # 1症例ずつ Dice を計算（バッチ内で混ぜると empty class 判定がずれるため）
        batch_size = images.shape[0]
        for i in range(batch_size):
            dice = dice_score(
                logits[i : i + 1], labels[i : i + 1], num_classes=num_classes
            )
            for key, value in dice.items():
                if value != value:  # NaN をスキップ（GT にも予測にも存在しないクラス）
                    continue
                if key not in all_dice:
                    all_dice[key] = []
                all_dice[key].append(value)

    # 平均（有効値のみ）
    avg_dice = {k: sum(v) / len(v) for k, v in all_dice.items() if v}
    return avg_dice


def train_step(
    config: Config,
    step: int,
    resume_path: str = None,
):
    """
    1つの学習ステップを実行
    
    Args:
        config: 設定
        step: ステップ番号（1 or 2）
        resume_path: 前ステップのチェックポイントパス（Step2の場合）
    """
    device = config.device
    set_seed(config.seed)

    # --- モデル構築 ---
    print("=" * 60)
    print(f"Step {step} 学習開始")
    print("=" * 60)

    model = SwinUNETRMoE(config).to(device)

    # ステップ設定
    if step == 1:
        text_desc = config.data.step1_text_description
        num_classes = config.data.step1_num_classes
        epochs = config.train.step1_epochs
        lr = config.train.step1_lr
        wd = config.train.step1_weight_decay

        # Step1 は事前学習チェックポイントからの resume があれば読み込む
        if resume_path is not None:
            load_checkpoint(model, resume_path, device=device)

    elif step == 2:
        # Step2 の重みロード順序が重要:
        #   1. prepare_step(1) で Expert0 と seg_head の構造を作ってから
        #   2. load_checkpoint() で Step1 の学習済み重みを復元する
        #
        # 逆順にすると experts_A_i が空 ParameterList のまま load されて
        # LoRA 重み / gating / seg_head が全て無視される（strict=False でも不一致）。
        model.prepare_step(
            step=1,
            text_description=config.data.step1_text_description,
            num_classes=config.data.step1_num_classes,
            device=device,
        )
        if resume_path is not None:
            load_checkpoint(model, resume_path, device=device)

        text_desc = config.data.step2_text_description
        num_classes = config.data.total_num_classes
        epochs = config.train.step2_epochs
        lr = config.train.step2_lr
        wd = config.train.step2_weight_decay

    else:
        raise ValueError(f"未対応のステップ: {step}")

    # 現在のステップを準備
    model.prepare_step(
        step=step,
        text_description=text_desc,
        num_classes=num_classes,
        device=device,
    )

    # パラメータ数の表示
    param_counts = model.count_trainable_params()
    print(f"パラメータ数:")
    for key, count in param_counts.items():
        print(f"  {key}: {count:,} ({count/1e6:.2f}M)")

    # --- データローダー ---
    train_loader = get_dataloader(config, step=step, is_train=True)
    val_loader   = get_dataloader(config, step=step, is_train=False)

    # --- 最適化 ---
    trainable_params = model.get_trainable_params()
    optimizer = AdamW(
        trainable_params,
        lr=lr,
        weight_decay=wd,
    )
    scheduler = build_scheduler(optimizer, epochs, config.train.warmup_epochs)
    criterion = DiceCELoss(num_classes=num_classes)
    # bf16 は GradScaler 不要（fp32 と同じ指数範囲のためスケーリング不要）
    # fp16 のみ GradScaler を使用
    use_scaler = config.use_amp and config.amp_dtype == "fp16"
    scaler = GradScaler() if use_scaler else None

    # --- ログ ---
    logger = TrainingLogger(log_dir=os.path.join(config.train.checkpoint_dir, "logs"))

    # --- 学習ループ ---
    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        # 学習
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, step, config.use_amp, config.amp_dtype,
        )
        scheduler.step()

        # ログ表示
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Step {step}] Epoch {epoch}/{epochs} | "
              f"Loss: {train_loss:.4f} | LR: {current_lr:.6f}")

        # 検証（一定間隔）
        if epoch % 10 == 0 or epoch == epochs:
            val_dice = validate(model, val_loader, num_classes, device)
            mean_dice = val_dice.get("mean", 0.0)

            print(f"  → Val Dice (mean): {mean_dice:.4f}")
            for cls_key, cls_dice in val_dice.items():
                if cls_key != "mean":
                    print(f"    {cls_key}: {cls_dice:.4f}")

            logger.log(epoch, step, {
                "train_loss": train_loss,
                "val_dice_mean": mean_dice,
                "lr": current_lr,
                **{f"val_{k}": v for k, v in val_dice.items()},
            })

            # ベストモデルの保存
            if mean_dice > best_dice:
                best_dice = mean_dice
                save_checkpoint(
                    model, optimizer, epoch, step,
                    {"best_dice": best_dice},
                    config.train.checkpoint_dir,
                    filename=f"step{step}_best.pth",
                )

        # 定期保存
        if epoch % config.train.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, step,
                {"train_loss": train_loss},
                config.train.checkpoint_dir,
                filename=f"step{step}_epoch{epoch}.pth",
            )

    print(f"\n[Step {step} 完了] Best Dice: {best_dice:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Low-Rank MoE Continual Learning")
    parser.add_argument("--step", type=int, required=True, choices=[1, 2],
                        help="学習ステップ (1 or 2)")
    parser.add_argument("--resume", type=str, default=None,
                        help="前ステップのチェックポイントパス")
    parser.add_argument("--config", type=str, default=None,
                        help="設定ファイルのパス（JSON）")

    # 主要パラメータのCLIオーバーライド
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--text_desc", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # 設定の読み込みと上書き
    config = Config()
    config.device = args.device

    if args.lora_rank is not None:
        config.model.lora_rank = args.lora_rank

    if args.step == 1:
        if args.epochs: config.train.step1_epochs = args.epochs
        if args.lr: config.train.step1_lr = args.lr
        if args.batch_size: config.train.step1_batch_size = args.batch_size
        if args.train_dir: config.data.step1_train_dir = args.train_dir
        if args.val_dir: config.data.step1_val_dir = args.val_dir
        if args.num_classes: config.data.step1_num_classes = args.num_classes
        if args.text_desc: config.data.step1_text_description = args.text_desc
    elif args.step == 2:
        if args.epochs: config.train.step2_epochs = args.epochs
        if args.lr: config.train.step2_lr = args.lr
        if args.batch_size: config.train.step2_batch_size = args.batch_size
        if args.train_dir: config.data.step2_train_dir = args.train_dir
        if args.val_dir: config.data.step2_val_dir = args.val_dir
        if args.num_classes:
            config.data.step2_num_classes = args.num_classes
            config.data.total_num_classes = config.data.step1_num_classes + args.num_classes - 1

    # 学習実行
    train_step(config, step=args.step, resume_path=args.resume)


if __name__ == "__main__":
    main()
