import os
import random
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

class DicomPairDataset(Dataset):
    def __init__(self, hr_root_dir, lr_root_dir, patch_size_hr=None, scale=None):
        """
        初期化処理：HRおよびLR画像のDICOMファイルパスを収集し、同じサブディレクトリ内からランダムに選べるようマッピングする
        :param hr_root_dir: HR画像が入っている大元のディレクトリパス
        :param lr_root_dir: LR画像が入っている大元のディレクトリパス
        """
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

        # LR: 100x100 のランダムクロップ
        x_lr = random.randint(0, lr_w - 100)
        y_lr = random.randint(0, lr_h - 100)
        lr_crop = lr_image[y_lr : y_lr + 100, x_lr : x_lr + 100]

        # HR: 200x200 のランダムクロップ
        x_hr = random.randint(0, hr_w - 200)
        y_hr = random.randint(0, hr_h - 200)
        hr_crop = hr_image[y_hr : y_hr + 200, x_hr : x_hr + 200]

        # 正規化 (0〜1 スケーリング)
        hr_crop = (hr_crop + 2048.0) / 6143.0
        lr_crop = (lr_crop + 2048.0) / 6143.0

        # テンソルに変換しチャネル次元追加
        hr_tensor = torch.from_numpy(hr_crop).unsqueeze(0)
        lr_tensor = torch.from_numpy(lr_crop).unsqueeze(0)

        return hr_tensor, lr_tensor
