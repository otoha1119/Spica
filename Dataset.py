import os
import random
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

class DicomPairDataset(Dataset):
    def __init__(self, hr_root_dir: str, lr_root_dir: str):
        """
        初期化処理：HRおよびLR画像のDICOMファイルパスを収集する
        :param hr_root_dir: HR画像が入っているpass1のディレクトリパス
        :param lr_root_dir: LR画像が入っているpass2のディレクトリパス
        """
        self.hr_files = []
        # hr_root_dir 以下すべての階層から .dcm ファイルを再帰的に収集
        for root, _, files in os.walk(hr_root_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    self.hr_files.append(os.path.join(root, file))

        self.lr_files = []
        # lr_root_dir 以下すべての階層から .dcm ファイルを再帰的に収集
        for root, _, files in os.walk(lr_root_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    self.lr_files.append(os.path.join(root, file))

    def __len__(self):
        # データセットの長さは仮に5000とします（エポックサイズの目安として使用）
        return 5000

    def __getitem__(self, index):
        # ランダムにHRとLR画像ファイルを1枚ずつ選択
        hr_path = random.choice(self.hr_files)
        lr_path = random.choice(self.lr_files)

        # DICOMファイルを読み込み、画像データをfloat32として取得
        hr_dicom = pydicom.dcmread(hr_path)
        lr_dicom = pydicom.dcmread(lr_path)
        hr_image = hr_dicom.pixel_array.astype(np.float32)
        lr_image = lr_dicom.pixel_array.astype(np.float32)

        # HRとLRの画像サイズ取得（想定：HR=1024x1024, LR=512x512）
        hr_h, hr_w = hr_image.shape
        lr_h, lr_w = lr_image.shape

        # 【変更】LR画像から100x100のランダムクロップ位置を決定
        x_lr = random.randint(0, lr_w - 100)
        y_lr = random.randint(0, lr_h - 100)
        # LRクロップ実施
        lr_crop = lr_image[y_lr : y_lr + 100, x_lr : x_lr + 100]

        # 【変更】HR画像から200x200のランダムクロップ位置を別に決定
        x_hr = random.randint(0, hr_w - 200)
        y_hr = random.randint(0, hr_h - 200)
        # HRクロップ実施
        hr_crop = hr_image[y_hr : y_hr + 200, x_hr : x_hr + 200]

        # 正規化処理：(値 + 2048) / 6143 によっておおよそ0〜1にスケーリング
        hr_crop = (hr_crop + 2048.0) / 6143.0
        lr_crop = (lr_crop + 2048.0) / 6143.0

        # PyTorchテンソルに変換し、チャネル次元を追加（[1, H, W]）
        hr_tensor = torch.from_numpy(hr_crop).unsqueeze(0)
        lr_tensor = torch.from_numpy(lr_crop).unsqueeze(0)

        # HRとLRそれぞれのクロップ座標も返す
        return hr_tensor, lr_tensor
