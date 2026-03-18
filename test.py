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
    per_sample_results = []  # [{"sample_id": str, "mean": float, cls_key: float, ...}]

    for i, batch_data in enumerate(dataloader):
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        # サンプルIDの取得（MONAIメタデータがあればファイル名、なければインデックス）
        meta = batch_data.get("image_meta_dict", {})
        filename_or_obj = meta.get("filename_or_obj", None)
        if filename_or_obj is not None:
            # バッチサイズ1想定。リストの場合は先頭要素を使用
            if isinstance(filename_or_obj, (list, tuple)):
                filename_or_obj = filename_or_obj[0]
            sample_id = os.path.basename(str(filename_or_obj))
        else:
            sample_id = f"sample_{i + 1:04d}"

        # テスト時はtraining_step=Noneでルーティング使用
        logits = model(images, training_step=None)
        dice = dice_score(logits, labels, num_classes=num_classes)

        result = {"sample_id": sample_id}
        result.update({k: float(v) for k, v in dice.items()})
        per_sample_results.append(result)

    # サンプルごとの結果をターミナルに表示
    cls_keys = sorted(k for k in per_sample_results[0].keys() if k not in ("sample_id", "mean"))
    header = f"{'sample_id':<40} {'mean_dice':>10}" + "".join(f"  {k:>12}" for k in cls_keys)
    print(f"\n{'='*len(header)}")
    print(f"[{dataset_name}] サンプル別評価結果")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for r in per_sample_results:
        row = f"{r['sample_id']:<40} {r.get('mean', 0.0):>10.4f}"
        row += "".join(f"  {r.get(k, 0.0):>12.4f}" for k in cls_keys)
        print(row)

    # 平均の表示
    avg_dice = {"mean": sum(r.get("mean", 0.0) for r in per_sample_results) / len(per_sample_results)}
    for k in cls_keys:
        avg_dice[k] = sum(r.get(k, 0.0) for r in per_sample_results) / len(per_sample_results)

    print("-" * len(header))
    avg_row = f"{'[平均]':<40} {avg_dice['mean']:>10.4f}"
    avg_row += "".join(f"  {avg_dice[k]:>12.4f}" for k in cls_keys)
    print(avg_row)

    return per_sample_results, avg_dice


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

    all_per_sample = []  # (dataset_name, per_sample_results) のリスト
    all_avg = []         # (dataset_name, avg_dice) のリスト

    # Step1データセットの評価
    if args.step1_test_dir:
        step1_loader = get_dataloader(
            data_dir=args.step1_test_dir,
            batch_size=1,
            is_train=False,
        )
        per_sample, avg_dice = evaluate_dataset(
            model, step1_loader,
            num_classes=config.data.total_num_classes,
            device=args.device,
            dataset_name=config.data.step1_name,
        )
        all_per_sample.append((config.data.step1_name, per_sample))
        all_avg.append((config.data.step1_name, avg_dice))

    # Step2データセットの評価
    if args.step2_test_dir:
        step2_loader = get_dataloader(
            data_dir=args.step2_test_dir,
            batch_size=1,
            is_train=False,
        )
        per_sample, avg_dice = evaluate_dataset(
            model, step2_loader,
            num_classes=config.data.total_num_classes,
            device=args.device,
            dataset_name=config.data.step2_name,
        )
        all_per_sample.append((config.data.step2_name, per_sample))
        all_avg.append((config.data.step2_name, avg_dice))

    # CSVへの保存（サンプル別 + 平均行）
    if all_per_sample:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        cls_keys = sorted(
            k for k in all_per_sample[0][1][0].keys()
            if k not in ("sample_id", "mean")
        )
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "sample_id", "mean_dice"] + cls_keys)
            for dataset_name, per_sample in all_per_sample:
                for r in per_sample:
                    row = [dataset_name, r["sample_id"], f"{r.get('mean', 0.0):.4f}"]
                    row += [f"{r.get(k, 0.0):.4f}" for k in cls_keys]
                    writer.writerow(row)
            # 平均行
            for dataset_name, avg_dice in all_avg:
                row = [dataset_name, "[平均]", f"{avg_dice.get('mean', 0.0):.4f}"]
                row += [f"{avg_dice.get(k, 0.0):.4f}" for k in cls_keys]
                writer.writerow(row)
        print(f"\n結果を保存しました: {args.output_csv}")


if __name__ == "__main__":
    main()
