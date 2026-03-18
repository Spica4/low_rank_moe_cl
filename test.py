"""
テストスクリプト: Class-level Continual Learning の評価

テスト時の動作:
1. 全ステップのエキスパートをロード
2. CLIPテキストembeddingからTop-1ハードルーティングで各トークンを振り分け
3. Step1とStep2のデータセットそれぞれでDice Scoreを計算

使い方:
  python test.py --checkpoint checkpoints/step2_best.pth \
                 --step1_test_dir /path/to/step1/test \
                 --step2_test_dir /path/to/step2/test
"""
import argparse
import csv
import os
import torch
from config import Config
from models.swin_unetr_moe import SwinUNETRMoE
from data.dataset import get_dataloader
from utils import dice_score, load_checkpoint


@torch.no_grad()
def evaluate_dataset(
    model: SwinUNETRMoE,
    dataloader,
    num_classes: int,
    device: str,
    dataset_name: str = "",
):
    """
    データセットごとの評価
    
    テスト時: ルーティングベース（training_step=None）
    → 各トークンがTop-1でエキスパートを自動選択
    """
    model.eval()
    all_dice = {}
    num_samples = 0

    for batch_data in dataloader:
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        # テスト時はtraining_step=Noneでルーティング使用
        logits = model(images, training_step=None)
        dice = dice_score(logits, labels, num_classes=num_classes)

        for key, value in dice.items():
            if key not in all_dice:
                all_dice[key] = []
            all_dice[key].append(value)
        num_samples += images.shape[0]

    # 平均
    avg_dice = {k: sum(v) / len(v) for k, v in all_dice.items()}

    print(f"\n{'='*50}")
    print(f"[{dataset_name}] 評価結果 (サンプル数: {num_samples})")
    print(f"{'='*50}")
    print(f"  Mean Dice: {avg_dice.get('mean', 0.0):.4f}")
    for cls_key in sorted(avg_dice.keys()):
        if cls_key != "mean":
            print(f"  {cls_key}: {avg_dice[cls_key]:.4f}")

    return avg_dice, num_samples


def main():
    parser = argparse.ArgumentParser(description="Low-Rank MoE CL - テスト")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="チェックポイントパス")
    parser.add_argument("--step1_test_dir", type=str, default=None,
                        help="Step1テストデータのディレクトリ")
    parser.add_argument("--step2_test_dir", type=str, default=None,
                        help="Step2テストデータのディレクトリ")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_csv", type=str, default="results.csv",
                        help="結果を保存するCSVファイルパス")
    args = parser.parse_args()

    config = Config()
    config.device = args.device

    # モデル構築と重み読み込み
    model = SwinUNETRMoE(config).to(args.device)

    # Step1, Step2のエキスパートを準備
    model.prepare_step(
        step=1,
        text_description=config.data.step1_text_description,
        num_classes=config.data.step1_num_classes,
        device=args.device,
    )
    model.prepare_step(
        step=2,
        text_description=config.data.step2_text_description,
        num_classes=config.data.total_num_classes,
        device=args.device,
    )

    # チェックポイント読み込み
    load_checkpoint(model, args.checkpoint, device=args.device)

    all_results = []

    # Step1データセットの評価
    if args.step1_test_dir:
        step1_loader = get_dataloader(
            data_dir=args.step1_test_dir,
            batch_size=1,
            is_train=False,
        )
        avg_dice, num_samples = evaluate_dataset(
            model, step1_loader,
            num_classes=config.data.total_num_classes,
            device=args.device,
            dataset_name=config.data.step1_name,
        )
        all_results.append((config.data.step1_name, num_samples, avg_dice))

    # Step2データセットの評価
    if args.step2_test_dir:
        step2_loader = get_dataloader(
            data_dir=args.step2_test_dir,
            batch_size=1,
            is_train=False,
        )
        avg_dice, num_samples = evaluate_dataset(
            model, step2_loader,
            num_classes=config.data.total_num_classes,
            device=args.device,
            dataset_name=config.data.step2_name,
        )
        all_results.append((config.data.step2_name, num_samples, avg_dice))

    # CSVへの保存
    if all_results:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        cls_keys = sorted(k for k in all_results[0][2].keys() if k != "mean")
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "num_samples", "mean_dice"] + cls_keys)
            for dataset_name, num_samples, avg_dice in all_results:
                row = [dataset_name, num_samples, f"{avg_dice.get('mean', 0.0):.4f}"]
                row += [f"{avg_dice.get(k, 0.0):.4f}" for k in cls_keys]
                writer.writerow(row)
        print(f"\n結果を保存しました: {args.output_csv}")


if __name__ == "__main__":
    main()
