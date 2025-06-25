# sdflow/Dataset.py (最終修正版)
import os
import random
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

class DicomPairDataset(Dataset):
    def __init__(self, hr_root_dir, lr_root_dir, patch_size_hr=128, scale=2, dequantize=False):
        self.hr_files = [os.path.join(path, name) for path, subdirs, files in os.walk(hr_root_dir) for name in files if name.lower().endswith('.dcm')]
        self.lr_files = [os.path.join(path, name) for path, subdirs, files in os.walk(lr_root_dir) for name in files if name.lower().endswith('.dcm')]

        self.patch_size_hr = patch_size_hr
        self.patch_size_lr = patch_size_hr // scale
        self.dequantize = dequantize

        print("-" * 50)
        print(f"[データセット情報]")
        print(f"  HR画像数: {len(self.hr_files)}, LR画像数: {len(self.lr_files)}")
        print(f"  HRパッチサイズ: {self.patch_size_hr}px")
        print(f"  LRパッチサイズ: {self.patch_size_lr}px")
        print(f"  非量子化(Dequantize): {self.dequantize}")
        print("-" * 50)

    def __len__(self):
        # 少ない方のデータセットサイズに合わせるなど、柔軟な設定が可能
        return min(len(self.hr_files), len(self.lr_files))

    def __getitem__(self, index):
        # indexはshuffleされるが、ここではランダム選択のため使わない
        hr_path = random.choice(self.hr_files)
        lr_path = random.choice(self.lr_files)

        hr_image = pydicom.dcmread(hr_path).pixel_array.astype(np.float32)
        lr_image = pydicom.dcmread(lr_path).pixel_array.astype(np.float32)

        hr_h, hr_w = hr_image.shape
        lr_h, lr_w = lr_image.shape

        # 引数で渡されたパッチサイズでランダムクロップ
        x_lr = random.randint(0, lr_w - self.patch_size_lr)
        y_lr = random.randint(0, lr_h - self.patch_size_lr)
        lr_crop = lr_image[y_lr : y_lr + self.patch_size_lr, x_lr : x_lr + self.patch_size_lr]

        x_hr = random.randint(0, hr_w - self.patch_size_hr)
        y_hr = random.randint(0, hr_h - self.patch_size_hr)
        hr_crop = hr_image[y_hr : y_hr + self.patch_size_hr, x_hr : x_hr + self.patch_size_hr]

        # 正規化
        hr_crop_normalized = (hr_crop + 1024) / 4095
        lr_crop_normalized = (lr_crop + 1024) / 4095

        hr_tensor = torch.from_numpy(hr_crop_normalized).unsqueeze(0)
        lr_tensor = torch.from_numpy(lr_crop_normalized).unsqueeze(0)

        if self.dequantize:
            hr_tensor += torch.rand_like(hr_tensor) / 255.0
            lr_tensor += torch.rand_like(lr_tensor) / 255.0

        hr_tensor = torch.clamp(hr_tensor, min=0, max=1)
        lr_tensor = torch.clamp(lr_tensor, min=0, max=1)

        return hr_tensor, lr_tensor
        #return hr_image, lr_image, hr_crop, lr_crop, hr_tensor, lr_tensor