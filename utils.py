"""
ユーティリティ関数
- Dice Score（論文の評価メトリクス）
- チェックポイント管理
- ログ
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import json
from datetime import datetime


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
    ignore_bg: bool = True,
) -> Dict[str, float]:
    """
    Dice Score (DSC) の計算
    
    論文の評価メトリクス: DSC = 2|P ∩ G| / (|P| + |G|)
    
    Args:
        pred: [batch, num_classes, H, W, D] (logits or softmax)
        target: [batch, 1, H, W, D] (integer labels)
        num_classes: クラス数
        smooth: スムージング係数
        ignore_bg: 背景クラスを無視するか
        
    Returns:
        dice_dict: クラスごとのDice Score と 平均
    """
    if pred.dim() == 5 and pred.shape[1] > 1:
        pred = pred.argmax(dim=1, keepdim=True)  # [batch, 1, H, W, D]

    target = target.long()
    if target.dim() == 5 and target.shape[1] == 1:
        target = target.squeeze(1)  # [batch, H, W, D]
    if pred.dim() == 5 and pred.shape[1] == 1:
        pred = pred.squeeze(1)

    dice_dict = {}
    start_class = 1 if ignore_bg else 0

    dice_values = []
    for c in range(start_class, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        # GT にも予測にも存在しないクラスは NaN とし、平均から除外する
        # (smooth で 1.0 になる誤カウントを防ぐ)
        if target_c.sum() == 0 and pred_c.sum() == 0:
            dice_dict[f"class_{c}"] = float("nan")
            continue

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_dict[f"class_{c}"] = dice.item()
        dice_values.append(dice.item())

    dice_dict["mean"] = sum(dice_values) / max(len(dice_values), 1)
    return dice_dict


class DiceLoss(nn.Module):
    """
    Dice Loss（学習用）
    
    Loss = 1 - DSC
    """

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-5,
        softmax: bool = True,
        ignore_bg: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.softmax = softmax
        self.ignore_bg = ignore_bg

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: [batch, num_classes, H, W, D] (logits)
        target: [batch, 1, H, W, D] (integer labels)
        """
        if self.softmax:
            pred = F.softmax(pred, dim=1)

        # One-hot encoding
        target_squeezed = target.squeeze(1).long()  # [batch, H, W, D]
        target_onehot = F.one_hot(
            target_squeezed, self.num_classes
        ).permute(0, 4, 1, 2, 3).float()  # [batch, num_classes, H, W, D]

        start_class = 1 if self.ignore_bg else 0
        loss = 0.0
        count = 0

        for c in range(start_class, self.num_classes):
            pred_c = pred[:, c]
            target_c = target_onehot[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss += 1.0 - dice
            count += 1

        return loss / max(count, 1)


class DiceCELoss(nn.Module):
    """Dice Loss + Cross Entropy Loss の組み合わせ"""

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
    ):
        super().__init__()
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_for_ce = target.squeeze(1).long()
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target_for_ce)
        return self.dice_weight * dice + self.ce_weight * ce


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    step: int,
    metrics: Dict,
    save_dir: str,
    filename: str = None,
):
    """チェックポイントの保存"""
    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        filename = f"step{step}_epoch{epoch}.pth"

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    path = os.path.join(save_dir, filename)
    torch.save(checkpoint, path)
    print(f"[チェックポイント保存] {path}")
    return path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer=None,
    device: str = "cuda",
):
    """チェックポイントの読み込み"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"[チェックポイント読み込み] {checkpoint_path}")
    print(f"  Step: {checkpoint.get('step', '?')}, "
          f"Epoch: {checkpoint.get('epoch', '?')}, "
          f"Metrics: {checkpoint.get('metrics', {})}")

    return checkpoint


class TrainingLogger:
    """学習ログの管理"""

    def __init__(self, log_dir: str = "./logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_file = os.path.join(
            log_dir,
            f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self.history = []

    def log(self, epoch: int, step: int, metrics: Dict):
        entry = {
            "epoch": epoch,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self.history.append(entry)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_best(self, metric_key: str = "val_dice_mean") -> Dict:
        if not self.history:
            return {}
        return max(self.history, key=lambda x: x.get(metric_key, 0))
