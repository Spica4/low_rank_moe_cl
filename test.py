"""
テストスクリプト: Class-level Continual Learning の評価

テスト時の動作:
1. 全ステップのエキスパートをロード
2. CLIPテキストembeddingからTop-1ハードルーティングで各トークンを振り分け
3. Step1とStep2のデータセットそれぞれでDice Scoreを計算
4. (オプション) Step1チェックポイントと比較して忘却率を計算

使い方:
  # Step2 単体評価
  python test.py --checkpoint checkpoints/step2_best.pth \
                 --step1_test_dir /path/to/step1/test \
                 --step2_test_dir /path/to/step2/test \
                 --output_csv results.csv

  # 忘却率を合わせて計算する場合
  python test.py --checkpoint checkpoints/step2_best.pth \
                 --step1_checkpoint checkpoints/step1_best.pth \
                 --step1_test_dir /path/to/step1/test \
                 --step2_test_dir /path/to/step2/test \
                 --output_csv results.csv
"""
import argparse
import csv
import math
import os
import torch
from monai.inferers import sliding_window_inference
from config import Config
from models.swin_unetr_moe import SwinUNETRMoE
from data.dataset import get_dataloader
from utils import dice_score, load_checkpoint


@torch.no_grad()
def evaluate_dataset(
    model: SwinUNETRMoE,
    dataloader,
    config: Config,
    num_classes: int,
    device: str,
    dataset_name: str = "",
    training_step: int = None,
):
    """
    データセットごとの評価

    Args:
        training_step: None = ルーティングベース、int = 指定エキスパート固定
    """
    model.eval()
    per_sample_results = []  # [{"sample_id": str, "mean": float, cls_key: float, ...}]

    for i, batch_data in enumerate(dataloader):
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        # サンプルIDの取得
        meta = batch_data.get("image_meta_dict", {})
        filename_or_obj = meta.get("filename_or_obj", None)
        if filename_or_obj is not None:
            if isinstance(filename_or_obj, (list, tuple)):
                filename_or_obj = filename_or_obj[0]
            sample_id = os.path.basename(str(filename_or_obj))
        else:
            sample_id = f"sample_{i + 1:04d}"

        # sliding window inference でフル画像を推論
        logits = sliding_window_inference(
            inputs=images,
            roi_size=config.data.spatial_size,
            sw_batch_size=4,
            predictor=lambda x: model(x, training_step=training_step),
            overlap=0.5,
        )
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
        row += "".join(f"  {r.get(k, float('nan')):>12.4f}" for k in cls_keys)
        print(row)

    # 平均の計算（NaN を除外）
    avg_dice = {}
    for k in cls_keys:
        vals = [r.get(k, float("nan")) for r in per_sample_results]
        valid = [v for v in vals if not math.isnan(v)]
        avg_dice[k] = sum(valid) / len(valid) if valid else float("nan")

    valid_means = [v for v in avg_dice.values() if not math.isnan(v)]
    avg_dice["mean"] = sum(valid_means) / len(valid_means) if valid_means else float("nan")

    print("-" * len(header))
    avg_row = f"{'[平均]':<40} {avg_dice['mean']:>10.4f}"
    avg_row += "".join(f"  {avg_dice.get(k, float('nan')):>12.4f}" for k in cls_keys)
    print(avg_row)

    return per_sample_results, avg_dice


def compute_forgetting(baseline_avg: dict, post_avg: dict, step1_classes: list) -> dict:
    """
    忘却率の計算

    forgetting_c = baseline_dice_c - post_cl_dice_c  (正値 = 忘却、負値 = 改善)

    Args:
        baseline_avg: Step1チェックポイントでStep1データを評価したときの平均Dice辞書
        post_avg:     Step2チェックポイントでStep1データを評価したときの平均Dice辞書
        step1_classes: Step1のクラスキーリスト (例: ["class_1", ..., "class_13"])

    Returns:
        forgetting: クラスごとの忘却率と平均
    """
    forgetting = {}
    valid_vals = []
    for k in step1_classes:
        b = baseline_avg.get(k, float("nan"))
        p = post_avg.get(k, float("nan"))
        if math.isnan(b) or math.isnan(p):
            forgetting[k] = float("nan")
        else:
            f = b - p
            forgetting[k] = f
            valid_vals.append(f)
    forgetting["mean"] = sum(valid_vals) / len(valid_vals) if valid_vals else float("nan")
    return forgetting


def print_forgetting_table(baseline_avg: dict, post_avg: dict, forgetting: dict,
                           step1_classes: list, dataset_name: str):
    """忘却率の比較テーブルを表示"""
    col_w = 12
    header = f"{'class':<15} {'baseline':>{col_w}} {'post_CL':>{col_w}} {'forgetting':>{col_w}}"
    print(f"\n{'='*len(header)}")
    print(f"[忘却率] {dataset_name} (Step1クラス)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for k in step1_classes:
        b = baseline_avg.get(k, float("nan"))
        p = post_avg.get(k, float("nan"))
        f = forgetting.get(k, float("nan"))
        b_str = f"{b:.4f}" if not math.isnan(b) else "  NaN"
        p_str = f"{p:.4f}" if not math.isnan(p) else "  NaN"
        f_str = f"{f:+.4f}" if not math.isnan(f) else "  NaN"
        print(f"{k:<15} {b_str:>{col_w}} {p_str:>{col_w}} {f_str:>{col_w}}")

    print("-" * len(header))
    bm = baseline_avg.get("mean", float("nan"))
    pm = post_avg.get("mean", float("nan"))
    fm = forgetting.get("mean", float("nan"))
    bm_str = f"{bm:.4f}" if not math.isnan(bm) else "  NaN"
    pm_str = f"{pm:.4f}" if not math.isnan(pm) else "  NaN"
    fm_str = f"{fm:+.4f}" if not math.isnan(fm) else "  NaN"
    print(f"{'[平均]':<15} {bm_str:>{col_w}} {pm_str:>{col_w}} {fm_str:>{col_w}}")


def build_step1_model(config: Config, device: str) -> SwinUNETRMoE:
    """Step1エキスパートのみを持つモデルを構築"""
    model = SwinUNETRMoE(config).to(device)
    model.prepare_step(
        step=1,
        text_description=config.data.step1_text_description,
        num_classes=config.data.step1_num_classes,
        device=device,
    )
    return model


def build_step2_model(config: Config, device: str) -> SwinUNETRMoE:
    """Step1 + Step2 エキスパートを持つモデルを構築"""
    model = SwinUNETRMoE(config).to(device)
    model.prepare_step(
        step=1,
        text_description=config.data.step1_text_description,
        num_classes=config.data.step1_num_classes,
        device=device,
    )
    model.prepare_step(
        step=2,
        text_description=config.data.step2_text_description,
        num_classes=config.data.total_num_classes,
        device=device,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Low-Rank MoE CL - テスト")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Step2チェックポイントパス (メイン評価用)")
    parser.add_argument("--step1_checkpoint", type=str, default=None,
                        help="Step1チェックポイントパス (忘却率計算用ベースライン)")
    parser.add_argument("--step1_test_dir", type=str, default=None,
                        help="Step1テストデータのディレクトリ")
    parser.add_argument("--step2_test_dir", type=str, default=None,
                        help="Step2テストデータのディレクトリ")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_csv", type=str, default="results.csv",
                        help="結果を保存するCSVファイルパス")
    parser.add_argument("--force_step1_expert", action="store_true",
                        help="Step1データ評価時に Expert0 を強制使用 (routing バイパス)"
                             " → ルーティング vs ヘッド破壊のどちらが主因か診断できる")
    args = parser.parse_args()

    config = Config()
    config.device = args.device

    # Step1クラスのキーリスト (class_1 〜 class_{step1_num_classes-1})
    step1_class_keys = [f"class_{c}" for c in range(1, config.data.step1_num_classes)]

    # =====================================================================
    # ベースライン評価: Step1チェックポイント × Step1テストデータ
    # =====================================================================
    baseline_avg_step1 = None
    baseline_per_sample_step1 = None

    if args.step1_checkpoint and args.step1_test_dir:
        print("\n" + "=" * 60)
        print("[ベースライン評価] Step1チェックポイントでStep1データを評価")
        print("=" * 60)
        model_base = build_step1_model(config, args.device)
        load_checkpoint(model_base, args.step1_checkpoint, device=args.device)

        config.data.step1_val_dir = args.step1_test_dir
        step1_loader = get_dataloader(config, step=1, is_train=False, use_cache=False)

        # Step1モデルは training_step=1 で step1エキスパートを固定使用
        baseline_per_sample_step1, baseline_avg_step1 = evaluate_dataset(
            model_base, step1_loader,
            config=config,
            num_classes=config.data.step1_num_classes,
            device=args.device,
            dataset_name=f"{config.data.step1_name} (baseline / step1 ckpt)",
            training_step=1,
        )
        del model_base
        torch.cuda.empty_cache()

    # =====================================================================
    # メイン評価: Step2チェックポイント
    # =====================================================================
    print("\n" + "=" * 60)
    print("[メイン評価] Step2チェックポイントで評価")
    print("=" * 60)
    model = build_step2_model(config, args.device)
    load_checkpoint(model, args.checkpoint, device=args.device)

    all_per_sample = []  # (dataset_name, per_sample_results) のリスト
    all_avg = []         # (dataset_name, avg_dice) のリスト

    post_avg_step1 = None
    post_per_sample_step1 = None

    # Step1データセットの評価（CL後）
    if args.step1_test_dir:
        config.data.step1_val_dir = args.step1_test_dir
        step1_loader = get_dataloader(config, step=1, is_train=False, use_cache=False)

        # --force_step1_expert: Expert0 強制使用でルーティングをバイパス
        # routing バイアスとヘッド破壊のどちらが主因かを切り分けるための診断オプション
        force_ts = 1 if args.force_step1_expert else None
        eval_name = (
            f"{config.data.step1_name} (post-CL / step2 ckpt / Expert0強制)"
            if args.force_step1_expert
            else f"{config.data.step1_name} (post-CL / step2 ckpt)"
        )
        post_per_sample_step1, post_avg_step1 = evaluate_dataset(
            model, step1_loader,
            config=config,
            num_classes=config.data.total_num_classes,
            device=args.device,
            dataset_name=eval_name,
            training_step=force_ts,
        )
        all_per_sample.append((config.data.step1_name, post_per_sample_step1))
        all_avg.append((config.data.step1_name, post_avg_step1))

    # Step2データセットの評価
    if args.step2_test_dir:
        config.data.step2_val_dir = args.step2_test_dir
        step2_loader = get_dataloader(config, step=2, is_train=False, use_cache=False)
        per_sample, avg_dice = evaluate_dataset(
            model, step2_loader,
            config=config,
            num_classes=config.data.total_num_classes,
            device=args.device,
            dataset_name=config.data.step2_name,
        )
        all_per_sample.append((config.data.step2_name, per_sample))
        all_avg.append((config.data.step2_name, avg_dice))

    # =====================================================================
    # 忘却率の計算と表示
    # =====================================================================
    forgetting = None
    if baseline_avg_step1 is not None and post_avg_step1 is not None:
        forgetting = compute_forgetting(baseline_avg_step1, post_avg_step1, step1_class_keys)
        print_forgetting_table(
            baseline_avg_step1, post_avg_step1, forgetting,
            step1_class_keys, config.data.step1_name,
        )

    # =====================================================================
    # CSVへの保存
    # =====================================================================
    if all_per_sample or baseline_per_sample_step1 is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

        # CSVのクラス列: step1クラス + step2以降のクラスを合わせたもの
        all_keys_set = set()
        if baseline_per_sample_step1:
            all_keys_set.update(
                k for k in baseline_per_sample_step1[0].keys()
                if k not in ("sample_id", "mean")
            )
        for _, ps in all_per_sample:
            if ps:
                all_keys_set.update(
                    k for k in ps[0].keys()
                    if k not in ("sample_id", "mean")
                )
        cls_keys = sorted(all_keys_set)

        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "sample_id", "mean_dice"] + cls_keys)

            # ベースライン行
            if baseline_per_sample_step1 is not None:
                tag = f"{config.data.step1_name}_baseline"
                for r in baseline_per_sample_step1:
                    row = [tag, r["sample_id"], f"{r.get('mean', float('nan')):.4f}"]
                    row += [
                        "" if math.isnan(r.get(k, float("nan"))) else f"{r.get(k, float('nan')):.4f}"
                        for k in cls_keys
                    ]
                    writer.writerow(row)
                # ベースライン平均行
                row = [tag, "[平均]", f"{baseline_avg_step1.get('mean', float('nan')):.4f}"]
                row += [
                    "" if math.isnan(baseline_avg_step1.get(k, float("nan")))
                    else f"{baseline_avg_step1.get(k, float('nan')):.4f}"
                    for k in cls_keys
                ]
                writer.writerow(row)

            # Post-CL行
            for dataset_name, per_sample in all_per_sample:
                tag = f"{dataset_name}_post_CL"
                for r in per_sample:
                    row = [tag, r["sample_id"], f"{r.get('mean', float('nan')):.4f}"]
                    row += [
                        "" if math.isnan(r.get(k, float("nan"))) else f"{r.get(k, float('nan')):.4f}"
                        for k in cls_keys
                    ]
                    writer.writerow(row)
            for dataset_name, avg_dice in all_avg:
                tag = f"{dataset_name}_post_CL"
                row = [tag, "[平均]", f"{avg_dice.get('mean', float('nan')):.4f}"]
                row += [
                    "" if math.isnan(avg_dice.get(k, float("nan")))
                    else f"{avg_dice.get(k, float('nan')):.4f}"
                    for k in cls_keys
                ]
                writer.writerow(row)

            # 忘却率行
            if forgetting is not None:
                tag = f"{config.data.step1_name}_forgetting"
                row = [tag, "[平均]", f"{forgetting.get('mean', float('nan')):+.4f}"]
                row += [
                    "" if math.isnan(forgetting.get(k, float("nan")))
                    else f"{forgetting.get(k, float('nan')):+.4f}"
                    for k in cls_keys
                ]
                writer.writerow(row)

        print(f"\n結果を保存しました: {args.output_csv}")


if __name__ == "__main__":
    main()
