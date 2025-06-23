import os
import random
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

class DicomPairDataset(Dataset):
    def __init__(self, hr_root_dir, lr_root_dir, patch_size_hr=None, scale=None):
        """
        :param hr_root_dir: HR画像が入っている大元のディレクトリパス
        :param lr_root_dir: LR画像が入っている大元のディレクトリパス
        """
        self.patch_size_hr = patch_size_hr
        self.patch_size_lr = patch_size_hr // scale if patch_size_hr and scale else 100
        
        print("-" * 50)
        print(f"[データセット情報]")
        print(f"  HRパッチサイズ: {self.patch_size_hr}px")
        print(f"  LRパッチサイズ: {self.patch_size_lr}px  ({self.patch_size_hr} // {scale} で計算)")
        print("-" * 50)
        
        # HR用: サブディレクトリごとにDICOMファイルを収集
        self.hr_files_by_dir = {}
        for root, dirs, files in os.walk(hr_root_dir):
            # そのディレクトリ内の .dcm ファイルのみをリスト化
            dcm_files = [os.path.join(root, f) for f in files if f.lower().endswith('.dcm')]
            if dcm_files:
                self.hr_files_by_dir[root] = dcm_files

        # LR用: 同様にサブディレクトリごとに収集
        self.lr_files_by_dir = {}
        for root, dirs, files in os.walk(lr_root_dir):
            dcm_files = [os.path.join(root, f) for f in files if f.lower().endswith('.dcm')]
            if dcm_files:
                self.lr_files_by_dir[root] = dcm_files

        # サブディレクトリ一覧を作成
        self.hr_dirs = list(self.hr_files_by_dir.keys())
        self.lr_dirs = list(self.lr_files_by_dir.keys())

        # エポックサイズを決める（仮に5000とする）
        self.dataset_size = 5000

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # HR側: ランダムにサブディレクトリを選び、その中からファイルを1つ選択
        hr_dir = random.choice(self.hr_dirs)
        hr_path = random.choice(self.hr_files_by_dir[hr_dir])
        lr_dir = random.choice(self.lr_dirs)
        lr_path = random.choice(self.lr_files_by_dir[lr_dir])

        # DICOM読み込み
        hr_dicom = pydicom.dcmread(hr_path)
        lr_dicom = pydicom.dcmread(lr_path)
        hr_image = hr_dicom.pixel_array.astype(np.float32)
        lr_image = lr_dicom.pixel_array.astype(np.float32)

        # 画像サイズ取得
        hr_h, hr_w = hr_image.shape
        lr_h, lr_w = lr_image.shape

        # LR: self.patch_size_lr を使ったランダムクロップ
        x_lr = random.randint(0, lr_w - self.patch_size_lr)
        y_lr = random.randint(0, lr_h - self.patch_size_lr)
        lr_crop = lr_image[y_lr : y_lr + self.patch_size_lr, x_lr : x_lr + self.patch_size_lr]

        # HR: self.patch_size_hr を使ったランダムクロップ
        x_hr = random.randint(0, hr_w - self.patch_size_hr)
        y_hr = random.randint(0, hr_h - self.patch_size_hr)
        hr_crop = hr_image[y_hr : y_hr + self.patch_size_hr, x_hr : x_hr + self.patch_size_hr]
        
        
        # 正規化 (0〜1 スケーリング) - 指定された固定値を使用
        # 正規化 (0〜1 スケーリング)
        hr_crop_normalized = (hr_crop + 1024) / 4095
        lr_crop_normalized = (lr_crop + 1024) / 4095
        
        # テンソルに変換
        hr_tensor = torch.from_numpy(hr_crop_normalized).unsqueeze(0)
        lr_tensor = torch.from_numpy(lr_crop_normalized).unsqueeze(0)
        
        # 0以下の値は0に、1を超える値は1にクリップ
        hr_tensor = torch.clamp(hr_tensor, min=0, max=1)
        lr_tensor = torch.clamp(lr_tensor, min=0, max=1)

        #return hr_tensor, lr_tensor
        return hr_image, lr_image, hr_crop, lr_crop, hr_tensor, lr_tensor