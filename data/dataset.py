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

        print(f"[データセットスキャン] {len(images)} 画像が見つかりました: {image_dir}")

        data = []
        for img_path in images:
            filename = os.path.basename(img_path)
            label_path = os.path.join(label_dir, filename)
            if os.path.exists(label_path):
                data.append({"image": img_path, "label": label_path})
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
    data_dir: str = None,
    data_list: List[Dict[str, str]] = None,
    spatial_size: Tuple[int, ...] = (96, 96, 96),
    batch_size: int = 3,
    num_workers: int = 4,
    is_train: bool = True,
    use_cache: bool = True,
) -> DataLoader:
    """
    データローダーを取得
    
    使用例:
        # ディレクトリから自動検出
        loader = get_dataloader(data_dir="/path/to/data", batch_size=3)
        
        # ファイルリストを直接指定
        data_list = [
            {"image": "/path/img1.nii.gz", "label": "/path/lbl1.nii.gz"},
            {"image": "/path/img2.nii.gz", "label": "/path/lbl2.nii.gz"},
        ]
        loader = get_dataloader(data_list=data_list, batch_size=3)
    """
    transforms = get_train_transforms(spatial_size) if is_train else get_val_transforms(spatial_size)

    if MONAI_AVAILABLE and use_cache and data_list is not None:
        dataset = CacheDataset(
            data=data_list,
            transform=transforms,
            cache_rate=1.0,
            num_workers=num_workers,
        )
    else:
        dataset = GenericMedicalDataset(
            data_dir=data_dir,
            data_list=data_list,
            transform=transforms,
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
