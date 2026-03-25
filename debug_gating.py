"""
debug_gating.py: ゲーティングスコアの可視化デバッグスクリプト

各レイヤーの GW_step0 (例: BTCV用) と GW_step1 (例: LiTS用) を収集し、
以下を可視化する:

  Figure 1: レイヤー別ルーティング比率
    - 各レイヤーで Expert0 / Expert1 に流れたトークンの割合 (%)
    - BTCVサンプルとLiTSサンプルで別々にプロット

  Figure 2: GW 値のヒストグラム
    - step0/step1 それぞれの GW 値の分布
    - BTCVサンプルとLiTSサンプルで比較

  Figure 3: レイヤー別 GW 平均値
    - 各レイヤーの mean(GW_step0) と mean(GW_step1) を比較

使い方:
  python debug_gating.py \
    --checkpoint checkpoints/step2_best.pth \
    --btcv_image /deeparea/sokabe/Dataset/BTCV/Abdomen/test/images/img0036.nii.gz \
    --lits_image /deeparea/sokabe/Dataset/LiTS/liver_tumor/test/images/volume-109.nii \
    --output_dir debug_gating_output
"""

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from models.language_gating import LanguageGuidedGating
from models.swin_unetr_moe import SwinUNETRMoE
from utils import load_checkpoint


# ---------------------------------------------------------------------------
# ゲーティング収集パッチ
# ---------------------------------------------------------------------------

# module_id -> List[Tensor] （呼ばれるたびに追記）
_gw_captures: dict[int, list] = defaultdict(list)    # sigmoid GW
_logit_captures: dict[int, list] = defaultdict(list)  # raw logit（softmax前）
_original_compute_gw = None
_original_compute_logit = None


def _patch_gating():
    """compute_gating_weights と compute_logits の両方をパッチして値を収集する。"""
    global _original_compute_gw, _original_compute_logit
    _original_compute_gw = LanguageGuidedGating.compute_gating_weights
    _original_compute_logit = LanguageGuidedGating.compute_logits

    def _debug_compute_gw(self, x, text_embedding):
        result = _original_compute_gw(self, x, text_embedding)
        _gw_captures[id(self)].append(result.detach().cpu().float())
        return result

    def _debug_compute_logit(self, x, text_embedding):
        result = _original_compute_logit(self, x, text_embedding)
        _logit_captures[id(self)].append(result.detach().cpu().float())
        return result

    LanguageGuidedGating.compute_gating_weights = _debug_compute_gw
    LanguageGuidedGating.compute_logits = _debug_compute_logit


def _restore_gating():
    """パッチを元に戻す。"""
    global _original_compute_gw, _original_compute_logit
    if _original_compute_gw is not None:
        LanguageGuidedGating.compute_gating_weights = _original_compute_gw
        _original_compute_gw = None
    if _original_compute_logit is not None:
        LanguageGuidedGating.compute_logits = _original_compute_logit
        _original_compute_logit = None


def _clear_captures():
    _gw_captures.clear()
    _logit_captures.clear()


# ---------------------------------------------------------------------------
# モデル構築
# ---------------------------------------------------------------------------

def build_model(config: Config, checkpoint_path: str, device: str) -> SwinUNETRMoE:
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
    load_checkpoint(model, checkpoint_path)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 画像の読み込みとパッチ生成
# ---------------------------------------------------------------------------

def load_and_crop(image_path: str, patch_size: tuple, device: str,
                  num_patches: int = 4) -> torch.Tensor:
    """
    NIfTI/MetaTensor をランダムクロップして [num_patches, 1, H, W, D] を返す。
    速度優先で SimpleITK ではなく MONAI の LoadImage を使用。
    """
    try:
        from monai.transforms import (
            LoadImage, EnsureChannelFirst, ScaleIntensityRange, RandSpatialCrop
        )
    except ImportError:
        raise ImportError("pip install monai が必要です")

    loader = LoadImage(image_only=True)
    add_ch = EnsureChannelFirst()
    scaler = ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)
    cropper = RandSpatialCrop(roi_size=patch_size, random_size=False)

    img = loader(image_path)
    img = add_ch(img)
    img = scaler(img)

    patches = []
    for _ in range(num_patches):
        patch = cropper(img)
        patches.append(torch.as_tensor(np.array(patch), dtype=torch.float32))

    return torch.stack(patches).to(device)  # [N, 1, H, W, D]


# ---------------------------------------------------------------------------
# 統計収集
# ---------------------------------------------------------------------------

def collect_stats(model: SwinUNETRMoE, patches: torch.Tensor):
    """
    パッチを推論してゲーティング統計を収集する。

    Returns:
        {
          "layer_key": {
              "gw_step0": np.ndarray,  # 全パッチ・全トークンの GW 値 (1D)
              "gw_step1": np.ndarray,
              "route_to_0": int,       # Expert0 に流れたトークン数
              "route_to_1": int,
          }
        }
    """
    # module_id → layer_key の逆引きマップ
    # step0 と step1 で同じ layer_key を共有するので step 番号も記録
    id_to_info: dict[int, dict] = {}
    for key, mod in model.gating_modules.items():
        # key: "step{i}_{layer_type}_{j}"
        parts = key.split("_", 1)   # ["step0", "ffn_3"] or ["step1", "attn_0"]
        step_idx = int(parts[0].replace("step", ""))
        layer_key = parts[1]
        id_to_info[id(mod)] = {"step_idx": step_idx, "layer_key": layer_key}

    _clear_captures()
    _patch_gating()

    try:
        with torch.no_grad():
            for patch in patches:
                model(patch.unsqueeze(0) if patch.dim() == 4 else patch)
    finally:
        _restore_gating()

    # --- GW と logit を layer_key × step_idx ごとに集約 ---
    layer_gws: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    layer_logits: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))

    for mod_id, gw_list in _gw_captures.items():
        if mod_id not in id_to_info:
            continue
        info = id_to_info[mod_id]
        step_idx = info["step_idx"]
        layer_key = info["layer_key"]
        for gw in gw_list:
            layer_gws[layer_key][step_idx].append(gw.numpy().flatten())

    for mod_id, logit_list in _logit_captures.items():
        if mod_id not in id_to_info:
            continue
        info = id_to_info[mod_id]
        step_idx = info["step_idx"]
        layer_key = info["layer_key"]
        for logit in logit_list:
            layer_logits[layer_key][step_idx].append(logit.numpy().flatten())

    # --- ルーティング決定を計算（softmax of logits） ---
    stats: dict[str, dict] = {}
    all_layer_keys = set(layer_gws.keys()) | set(layer_logits.keys())

    for layer_key in all_layer_keys:
        gw_dict = layer_gws.get(layer_key, {})
        logit_dict = layer_logits.get(layer_key, {})

        # GW（sigmoid）の取得
        if 0 in gw_dict and 1 in gw_dict:
            gw0 = np.concatenate(gw_dict[0])
            gw1 = np.concatenate(gw_dict[1])
            n = min(len(gw0), len(gw1))
            gw0, gw1 = gw0[:n], gw1[:n]
        else:
            continue

        # logit が取れた場合は softmax でルーティング確率を計算
        if 0 in logit_dict and 1 in logit_dict:
            l0 = np.concatenate(logit_dict[0])
            l1 = np.concatenate(logit_dict[1])
            n_l = min(len(l0), len(l1))
            l0, l1 = l0[:n_l], l1[:n_l]
            # softmax: exp(l) / (exp(l0) + exp(l1))
            exp0, exp1 = np.exp(l0 - np.maximum(l0, l1)), np.exp(l1 - np.maximum(l0, l1))
            denom = exp0 + exp1
            prob0, prob1 = exp0 / denom, exp1 / denom
            route_to_0 = int((prob0 >= prob1).sum())
            route_to_1 = int((prob1 >  prob0).sum())
            routing_source = "softmax_logit"
        else:
            # logit が取れない場合は GW の argmax にフォールバック
            route_to_0 = int((gw0 >= gw1).sum())
            route_to_1 = int((gw1 >  gw0).sum())
            prob0 = prob1 = None
            routing_source = "gw_argmax"

        stats[layer_key] = {
            "gw_step0": gw0,
            "gw_step1": gw1,
            "route_to_0": route_to_0,
            "route_to_1": route_to_1,
            "routing_source": routing_source,
        }
        if prob0 is not None:
            stats[layer_key]["prob_step0"] = prob0
            stats[layer_key]["prob_step1"] = prob1

    return stats


# ---------------------------------------------------------------------------
# プロット
# ---------------------------------------------------------------------------

def plot_routing_ratio(stats_btcv: dict, stats_lits: dict,
                       step_names: list, output_dir: str):
    """Figure 1: レイヤー別 Expert ルーティング比率 (%)"""
    layer_keys = sorted(stats_btcv.keys())
    x = np.arange(len(layer_keys))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Layer-wise Routing Ratio (% tokens per expert)", fontsize=13)

    for ax, stats, dataset_name in zip(axes,
                                       [stats_btcv, stats_lits],
                                       [step_names[0], step_names[1]]):
        ratio0, ratio1 = [], []
        for k in layer_keys:
            s = stats.get(k, {"route_to_0": 0, "route_to_1": 0})
            total = s["route_to_0"] + s["route_to_1"]
            ratio0.append(100.0 * s["route_to_0"] / total if total > 0 else 0.0)
            ratio1.append(100.0 * s["route_to_1"] / total if total > 0 else 0.0)

        bars0 = ax.bar(x - width / 2, ratio0, width,
                       label=f"Expert0 ({step_names[0]})", color="#4C72B0", alpha=0.85)
        bars1 = ax.bar(x + width / 2, ratio1, width,
                       label=f"Expert1 ({step_names[1]})", color="#DD8452", alpha=0.85)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Tokens (%)")
        ax.set_ylim(0, 110)
        ax.set_title(f"Input: {dataset_name} sample")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_keys, rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig1_routing_ratio.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  保存: {path}")


def plot_gw_histograms(stats_btcv: dict, stats_lits: dict,
                       step_names: list, output_dir: str):
    """Figure 2: GW 値のヒストグラム（全レイヤー統合）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Gating Weight (GW) Distributions (all layers combined)", fontsize=13)

    data = {
        (0, 0): (stats_btcv, "gw_step0", step_names[0], f"{step_names[0]} input / GW for Expert0"),
        (0, 1): (stats_btcv, "gw_step1", step_names[1], f"{step_names[0]} input / GW for Expert1"),
        (1, 0): (stats_lits, "gw_step0", step_names[0], f"{step_names[1]} input / GW for Expert0"),
        (1, 1): (stats_lits, "gw_step1", step_names[1], f"{step_names[1]} input / GW for Expert1"),
    }
    colors = {
        "gw_step0": "#4C72B0",
        "gw_step1": "#DD8452",
    }

    for (row, col), (stats, gw_key, _, title) in data.items():
        ax = axes[row, col]
        all_vals = np.concatenate([s[gw_key] for s in stats.values()]) if stats else np.array([])
        if len(all_vals) > 0:
            ax.hist(all_vals, bins=50, color=colors[gw_key], alpha=0.8, edgecolor="white")
            ax.axvline(all_vals.mean(), color="red", linestyle="--",
                       linewidth=1.2, label=f"mean={all_vals.mean():.4f}")
            ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("GW value (sigmoid output)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_gw_histograms.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  保存: {path}")


def plot_routing_prob_histograms(stats_btcv: dict, stats_lits: dict,
                                 step_names: list, output_dir: str):
    """Figure 4: Softmax ルーティング確率のヒストグラム（修正後の実際の競合を確認）"""
    # prob データが存在するレイヤーがあるか確認
    has_probs = any("prob_step0" in s for s in stats_btcv.values())
    if not has_probs:
        print("  [スキップ] routing prob データなし (logit が収集されていません)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Softmax Routing Probability Distributions (after fix)\n"
                 "理想: 各入力サンプルで対応Expertの確率が集中すべき", fontsize=12)

    data = {
        (0, 0): (stats_btcv, "prob_step0", f"{step_names[0]} input / P(Expert0={step_names[0]})"),
        (0, 1): (stats_btcv, "prob_step1", f"{step_names[0]} input / P(Expert1={step_names[1]})"),
        (1, 0): (stats_lits, "prob_step0", f"{step_names[1]} input / P(Expert0={step_names[0]})"),
        (1, 1): (stats_lits, "prob_step1", f"{step_names[1]} input / P(Expert1={step_names[1]})"),
    }
    colors = {"prob_step0": "#4C72B0", "prob_step1": "#DD8452"}
    ideal_label = {
        (0, 0): "← 高いと良い",  # BTCV入力にExpert0が高確率
        (0, 1): "← 低いと良い",  # BTCV入力にExpert1が低確率
        (1, 0): "← 低いと良い",  # LiTS入力にExpert0が低確率
        (1, 1): "← 高いと良い",  # LiTS入力にExpert1が高確率
    }
    ideal_good = {(0, 0): True, (0, 1): False, (1, 0): False, (1, 1): True}

    for (row, col), (stats, prob_key, title) in data.items():
        ax = axes[row, col]
        all_vals = np.concatenate([s[prob_key] for s in stats.values()
                                   if prob_key in s]) if stats else np.array([])
        if len(all_vals) > 0:
            color = colors[prob_key]
            ax.hist(all_vals, bins=50, color=color, alpha=0.8, edgecolor="white")
            ax.axvline(all_vals.mean(), color="red", linestyle="--",
                       linewidth=1.2, label=f"mean={all_vals.mean():.4f}")
            ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
            ax.legend(fontsize=8)

        bgcolor = "#e8f5e9" if ideal_good[(row, col)] else "#fce4ec"
        ax.set_facecolor(bgcolor)
        ax.set_title(f"{title}\n{ideal_label[(row, col)]}", fontsize=8)
        ax.set_xlabel("Routing probability (softmax output)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_routing_probs.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  保存: {path}")


def plot_layerwise_mean_gw(stats_btcv: dict, stats_lits: dict,
                           step_names: list, output_dir: str):
    """Figure 3: レイヤー別 GW 平均値の比較"""
    layer_keys = sorted(stats_btcv.keys())
    x = np.arange(len(layer_keys))
    width = 0.2

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Layer-wise Mean Gating Weight", fontsize=13)

    for ax, stats, dataset_name in zip(axes,
                                       [stats_btcv, stats_lits],
                                       [step_names[0], step_names[1]]):
        means0 = [stats[k]["gw_step0"].mean() if k in stats else 0.0 for k in layer_keys]
        means1 = [stats[k]["gw_step1"].mean() if k in stats else 0.0 for k in layer_keys]

        ax.bar(x - width / 2, means0, width,
               label=f"GW_step0 ({step_names[0]})", color="#4C72B0", alpha=0.85)
        ax.bar(x + width / 2, means1, width,
               label=f"GW_step1 ({step_names[1]})", color="#DD8452", alpha=0.85)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Mean GW")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Input: {dataset_name} sample")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_keys, rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_layerwise_mean_gw.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  保存: {path}")


# ---------------------------------------------------------------------------
# テキスト統計サマリー
# ---------------------------------------------------------------------------

def print_summary(stats: dict, dataset_name: str, step_names: list):
    print(f"\n{'='*70}")
    print(f"[{dataset_name}] ゲーティング統計サマリー")
    print(f"{'='*70}")
    print(f"{'layer':<15} {'mean GW0':>10} {'mean GW1':>10} "
          f"{'route→E0%':>11} {'route→E1%':>11}  {'勝者'}")
    print("-" * 70)

    total_to_0 = total_to_1 = 0
    for k in sorted(stats.keys()):
        s = stats[k]
        m0 = s["gw_step0"].mean()
        m1 = s["gw_step1"].mean()
        r0 = s["route_to_0"]
        r1 = s["route_to_1"]
        total = r0 + r1
        p0 = 100.0 * r0 / total if total > 0 else 0.0
        p1 = 100.0 * r1 / total if total > 0 else 0.0
        winner = f"E0({step_names[0]})" if m0 > m1 else f"E1({step_names[1]})"
        print(f"{k:<15} {m0:>10.4f} {m1:>10.4f} {p0:>10.1f}% {p1:>10.1f}%  {winner}")
        total_to_0 += r0
        total_to_1 += r1

    total = total_to_0 + total_to_1
    print("-" * 70)
    print(f"{'[全レイヤー合計]':<15} {'':>10} {'':>10} "
          f"{100.0*total_to_0/total:>10.1f}% {100.0*total_to_1/total:>10.1f}%")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ゲーティングスコア可視化")
    parser.add_argument("--checkpoint", required=True, help="Step2チェックポイントパス")
    parser.add_argument("--btcv_image", required=True,
                        help="BTCVテスト画像 (.nii.gz)")
    parser.add_argument("--lits_image", required=True,
                        help="LiTSテスト画像 (.nii.gz)")
    parser.add_argument("--num_patches", type=int, default=4,
                        help="1画像あたりのランダムクロップ数 (デフォルト: 4)")
    parser.add_argument("--output_dir", default="debug_gating_output",
                        help="出力ディレクトリ (デフォルト: debug_gating_output)")
    parser.add_argument("--device", default="cuda",
                        help="デバイス (デフォルト: cuda)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = Config()
    device = args.device if torch.cuda.is_available() else "cpu"
    step_names = [config.data.step1_name, config.data.step2_name]

    print(f"[モデル構築] チェックポイント: {args.checkpoint}")
    model = build_model(config, args.checkpoint, device)

    patch_size = config.data.spatial_size

    # --- BTCV サンプルで統計収集 ---
    print(f"\n[BTCV] 画像ロード + {args.num_patches}クロップで推論...")
    btcv_patches = load_and_crop(args.btcv_image, patch_size, device, args.num_patches)
    stats_btcv = collect_stats(model, btcv_patches)

    # --- LiTS サンプルで統計収集 ---
    print(f"[LiTS] 画像ロード + {args.num_patches}クロップで推論...")
    lits_patches = load_and_crop(args.lits_image, patch_size, device, args.num_patches)
    stats_lits = collect_stats(model, lits_patches)

    # --- テキスト統計 ---
    print_summary(stats_btcv, config.data.step1_name, step_names)
    print_summary(stats_lits, config.data.step2_name, step_names)

    # --- プロット ---
    print(f"\n[プロット生成] -> {args.output_dir}/")
    plot_routing_ratio(stats_btcv, stats_lits, step_names, args.output_dir)
    plot_gw_histograms(stats_btcv, stats_lits, step_names, args.output_dir)
    plot_layerwise_mean_gw(stats_btcv, stats_lits, step_names, args.output_dir)
    plot_routing_prob_histograms(stats_btcv, stats_lits, step_names, args.output_dir)

    print("\n[完了] 4ファイルを出力しました:")
    print(f"  {args.output_dir}/fig1_routing_ratio.png")
    print(f"  {args.output_dir}/fig2_gw_histograms.png")
    print(f"  {args.output_dir}/fig3_layerwise_mean_gw.png")
    print(f"  {args.output_dir}/fig4_routing_probs.png  ← softmax修正後のルーティング確率")


if __name__ == "__main__":
    main()
