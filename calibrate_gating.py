"""
calibrate_gating.py: Step0 ゲーティングのキャリブレーション

[問題]
  Step1 学習時は競合エキスパートが存在しないため、step0 ゲーティングの
  text_proj は「GW ≈ 0.5 (logit ≈ 0)」を出力したまま収束する。
  Step2 学習後のテスト時に logit_step0 ≈ 0 < logit_step1 となり、
  BTCV 入力でも Expert1 が支配的になる。

[修正内容]
  Step0 ゲーティングの text_proj のみを BTCV データで短期再訓練する。
  エキスパートの重み (LoRA) は一切変更しない。

  損失:
    BCE(GW_step0_all_layers_mean, target=0.8)
    ← BTCV 特徴量に対して GW_step0 が高くなるよう誘導

  期待効果:
    logit_step0 が BTCV 特徴量に対して正の値を持つようになり、
    競合時に Expert0 が正しく選ばれる。

使い方:
  python calibrate_gating.py \\
    --checkpoint checkpoints/step2_best.pth \\
    --btcv_train_dir /deeparea/sokabe/Dataset/BTCV/Abdomen/train \\
    --output checkpoints/step2_calibrated.pth \\
    --epochs 10
"""

import argparse
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from config import Config
from models.language_gating import LanguageGuidedGating
from models.swin_unetr_moe import SwinUNETRMoE
from data.dataset import get_dataloader
from utils import load_checkpoint


# ---------------------------------------------------------------------------
# モデル構築
# ---------------------------------------------------------------------------

def build_step2_model(config: Config, checkpoint_path: str, device: str) -> SwinUNETRMoE:
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
    return model


# ---------------------------------------------------------------------------
# キャリブレーション
# ---------------------------------------------------------------------------

def calibrate(model: SwinUNETRMoE, dataloader, device: str,
               epochs: int, lr: float, gw_target: float):
    """
    Step0 ゲーティング (text_proj) のみを BTCV データで再訓練する。

    実装:
      LanguageGuidedGating.forward_train をパッチして GW を収集し、
      BCE(GW, target) を損失として backprop する。
      Step0 の text_proj だけが更新される。
    """

    # ----- 凍結設定 -----
    for param in model.parameters():
        param.requires_grad = False

    gating_params = []
    for key, gating in model.gating_modules.items():
        if key.startswith("step0_"):
            for param in gating.parameters():
                param.requires_grad = True
                gating_params.append(param)

    if not gating_params:
        raise RuntimeError("step0 ゲーティングパラメータが見つかりません")

    n_params = sum(p.numel() for p in gating_params)
    print(f"  訓練対象パラメータ数: {n_params:,}  (step0 gating text_proj のみ)")

    optimizer = AdamW(gating_params, lr=lr, weight_decay=0.0)

    # ----- forward_train パッチ: GW を収集して loss に追加 -----
    # step0 の forward_train が呼ばれるたびに GW を収集するリスト
    _collected_gws: list = []
    _step0_ids = {id(g) for k, g in model.gating_modules.items() if k.startswith("step0_")}

    original_forward_train = LanguageGuidedGating.forward_train

    def _patched_forward_train(self, x, text_embedding):
        gw = self.compute_gating_weights(x, text_embedding)  # [B, N, 1]
        if id(self) in _step0_ids:
            _collected_gws.append(gw)
        return x * gw

    LanguageGuidedGating.forward_train = _patched_forward_train

    model._is_test_mode = False
    model.train()
    model._current_expert_idx = 0

    target = torch.tensor(gw_target, device=device)

    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            for batch_data in dataloader:
                images = batch_data["image"].to(device)
                _collected_gws.clear()

                optimizer.zero_grad()

                # training_step=1 固定で forward（Step0 エキスパートのみ使用）
                _ = model(images, training_step=1)

                if not _collected_gws:
                    continue

                # 全レイヤーの GW を結合して BCE 損失を計算
                # _collected_gws: List[Tensor[B, N, 1]]
                all_gws = torch.cat([g.reshape(-1) for g in _collected_gws])
                loss = F.binary_cross_entropy(all_gws, target.expand_as(all_gws))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(gating_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            # GW の平均値を計算して表示
            with torch.no_grad():
                sample_gws = _collected_gws
                if sample_gws:
                    mean_gw = torch.cat([g.reshape(-1) for g in sample_gws]).mean().item()
                else:
                    mean_gw = float("nan")
            print(f"  Epoch [{epoch:3d}/{epochs}]  BCE loss={avg_loss:.6f}  "
                  f"mean(GW_step0)={mean_gw:.4f}  (目標: {gw_target:.2f})")

    finally:
        LanguageGuidedGating.forward_train = original_forward_train

    # 全パラメータ再凍結
    for param in model.parameters():
        param.requires_grad = False


# ---------------------------------------------------------------------------
# 検証: GW の変化を確認
# ---------------------------------------------------------------------------

@torch.no_grad()
def verify_gw_distribution(model: SwinUNETRMoE, dataloader, device: str, n_batches: int = 3):
    """キャリブレーション後の GW_step0 / GW_step1 分布を確認する"""
    model.eval()
    model._is_test_mode = False
    model._current_expert_idx = 0

    gw0_all, gw1_all = [], []

    # step0 の forward_train にフックを追加
    _step0_ids = {id(g) for k, g in model.gating_modules.items() if k.startswith("step0_")}

    original_forward_train = LanguageGuidedGating.forward_train

    def _collect(self, x, text_embedding):
        gw = self.compute_gating_weights(x, text_embedding)
        if id(self) in _step0_ids:
            gw0_all.append(gw.cpu().float().reshape(-1))
        return x * gw

    LanguageGuidedGating.forward_train = _collect

    try:
        for i, batch_data in enumerate(dataloader):
            if i >= n_batches:
                break
            images = batch_data["image"].to(device)
            _ = model(images, training_step=1)
    finally:
        LanguageGuidedGating.forward_train = original_forward_train

    if gw0_all:
        gw0 = torch.cat(gw0_all).numpy()
        print(f"\n  [検証] GW_step0 (BTCV入力) after calibration:")
        print(f"    mean = {gw0.mean():.4f}  (0.5以上が理想)")
        print(f"    std  = {gw0.std():.4f}")
        print(f"    % > 0.5: {(gw0 > 0.5).mean() * 100:.1f}%  (高いほど Expert0 が勝ちやすい)")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step0 ゲーティング キャリブレーション")
    parser.add_argument("--checkpoint", required=True,
                        help="Step2 チェックポイント (.pth)")
    parser.add_argument("--btcv_train_dir", required=True,
                        help="BTCV 学習データディレクトリ")
    parser.add_argument("--output", required=True,
                        help="出力チェックポイントパス (.pth)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="キャリブレーションエポック数 (デフォルト: 10)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学習率 (デフォルト: 1e-3)")
    parser.add_argument("--gw_target", type=float, default=0.8,
                        help="BTCV 入力に対する GW_step0 の目標値 (デフォルト: 0.8)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = Config()
    device = args.device if torch.cuda.is_available() else "cpu"

    print("[モデル構築 & チェックポイントロード]")
    model = build_step2_model(config, args.checkpoint, device)

    print(f"\n[データロード] {args.btcv_train_dir}")
    dataloader = get_dataloader(
        data_dir=args.btcv_train_dir,
        batch_size=config.train.step1_batch_size,
        spatial_size=config.data.spatial_size,
        num_classes=config.data.step1_num_classes,
        num_workers=2,
        is_train=True,
    )

    print(f"\n[キャリブレーション開始]")
    print(f"  エポック数 : {args.epochs}")
    print(f"  学習率     : {args.lr}")
    print(f"  GW 目標値  : {args.gw_target}")
    print(f"  ※ LoRA エキスパート重みは変更しません\n")

    calibrate(model, dataloader, device,
               epochs=args.epochs, lr=args.lr, gw_target=args.gw_target)

    print("\n[キャリブレーション後の GW 確認]")
    verify_gw_distribution(model, dataloader, device)

    print(f"\n[チェックポイント保存] {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, args.output)

    print("\n[完了]")
    print("次のステップ:")
    print(f"  python debug_gating.py --checkpoint {args.output} \\")
    print(f"    --btcv_image <btcv_test.nii.gz> --lits_image <lits_test.nii.gz>")
    print(f"  → fig1 の BTCV 行で Expert0 が増加しているか確認")


if __name__ == "__main__":
    main()
