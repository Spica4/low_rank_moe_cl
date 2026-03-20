"""
データセットモジュール（差し替え可能）

このファイルを編集して、自分のデータセットに合わせてください。
MONAI Transformsを使った3D医療画像の前処理パイプラインを提供します。

カスタムデータセットの使い方:
1. get_dataloader() の data_dir と transform を自分のデータに合わせて変更
2. または GenericMedicalDataset を継承して独自のDatasetを作成
"""
import os
import glob
from typing import Optional, List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        ScaleIntensityRanged,
        CropForegroundd,
        SpatialPadd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandRotate90d,
        ToTensord,
        Resized,
    )
    from monai.data import CacheDataset, DataLoader as MonaiDataLoader
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("[警告] MONAIがインストールされていません。基本的なDatasetを使用します。")


class GenericMedicalDataset(Dataset):
    """
    汎用的な医療画像データセット
    
    ディレクトリ構造の例:
    data_dir/
        images/
            img_001.nii.gz
            img_002.nii.gz
            ...
        labels/
            img_001.nii.gz
            img_002.nii.gz
            ...
    
    または、辞書のリストとして直接データを渡すことも可能:
    data_list = [
        {"image": "/path/to/img1.nii.gz", "label": "/path/to/label1.nii.gz"},
        ...
    ]
    """

    def __init__(
        self,
        data_dir: str = None,
        data_list: List[Dict[str, str]] = None,
        transform=None,
        image_key: str = "image",
        label_key: str = "label",
    ):
        super().__init__()
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

        if data_list is not None:
            self.data = data_list
        elif data_dir is not None:
            self.data = self._scan_directory(data_dir)
        else:
            raise ValueError("data_dir または data_list を指定してください")

    def _scan_directory(self, data_dir: str) -> List[Dict[str, str]]:
        """ディレクトリからファイルリストを自動生成"""
        image_dir = os.path.join(data_dir, "images")
        label_dir = os.path.join(data_dir, "labels")

        if not os.path.exists(image_dir):
            # フラットなディレクトリ構造の場合
            images = sorted(
                glob.glob(os.path.join(data_dir, "*.nii.gz")) +
                glob.glob(os.path.join(data_dir, "*.nii"))
            )
            return [{"image": img, "label": img.replace("image", "label")} for img in images]

        images = sorted(
            glob.glob(os.path.join(image_dir, "*.nii.gz")) +
            glob.glob(os.path.join(image_dir, "*.nii"))
        )

        labels = sorted(
            glob.glob(os.path.join(label_dir, "*.nii.gz")) +
            glob.glob(os.path.join(label_dir, "*.nii"))
        )

        print(f"[データセットスキャン] {len(images)} 画像が見つかりました: {image_dir}")
        print(f"[データセットスキャン] {len(labels)} ラベルが見つかりました: {label_dir}")

        data = []
        # for img_path in images:
        #     filename = os.path.basename(img_path)
        #     label_path = os.path.join(label_dir, filename)
        #     if os.path.exists(label_path):
        #         data.append({"image": img_path, "label": label_path})

        for i in range(len(images)):
            img_path = images[i]
            label_path = labels[i] if i < len(labels) else None
            if os.path.exists(label_path):
                data.append({"image": img_path, "label": label_path})
            else:
                print(f"[警告] 対応するラベルが見つかりません: {label_path}")
        
        print(f"[データセットスキャン] {len(data)} ペアが見つかりました: {label_dir}")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform is not None:
            item = self.transform(item)
        return item


def get_train_transforms(spatial_size: Tuple[int, ...] = (96, 96, 96)):
    """学習用の前処理パイプライン"""
    if not MONAI_AVAILABLE:
        return None

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # クロップサイズより小さい画像をパディング（LiTS 等で z 方向が薄い場合に対応）
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=1, neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms(spatial_size: Tuple[int, ...] = (96, 96, 96)):
    """検証用の前処理パイプライン"""
    if not MONAI_AVAILABLE:
        return None

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ])


def get_dataloader(
    config,
    step: int,
    is_train: bool = True,
    use_cache: bool = True,
) -> DataLoader:
    """
    Config と step 番号からデータローダーを取得

    使用例:
        train_loader = get_dataloader(config, step=1, is_train=True)
        val_loader   = get_dataloader(config, step=1, is_train=False)
    """
    data_cfg = config.data
    train_cfg = config.train

    if step == 1:
        data_dir   = data_cfg.step1_train_dir if is_train else data_cfg.step1_val_dir
        batch_size = train_cfg.step1_batch_size if is_train else 1
    elif step == 2:
        data_dir   = data_cfg.step2_train_dir if is_train else data_cfg.step2_val_dir
        batch_size = train_cfg.step2_batch_size if is_train else 1
    else:
        raise ValueError(f"未対応のステップ: {step}")

    spatial_size = tuple(data_cfg.spatial_size)
    num_workers  = data_cfg.num_workers

    transforms = get_train_transforms(spatial_size) if is_train else get_val_transforms(spatial_size)

    if MONAI_AVAILABLE and use_cache:
        data_list = GenericMedicalDataset(data_dir=data_dir)._scan_directory(data_dir)
        if len(data_list) == 0:
            raise ValueError(
                f"データが見つかりませんでした: {data_dir}\n"
                f"config.data.step{step}_{'train' if is_train else 'val'}_dir を確認してください。"
            )
        dataset = CacheDataset(
            data=data_list,
            transform=transforms,
            cache_rate=1.0,
            num_workers=num_workers,
        )
    else:
        dataset = GenericMedicalDataset(data_dir=data_dir, transform=transforms)
        if len(dataset) == 0:
            raise ValueError(
                f"データが見つかりませんでした: {data_dir}\n"
                f"config.data.step{step}_{'train' if is_train else 'val'}_dir を確認してください。"
            )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    return loader
