###############################################################
# utils.py
###############################################################
import os
import torch
import torchvision.utils as vutils


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_image_tensor(tensor, filepath):
    # tensor: (B, C, H, W) in [0,1]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    vutils.save_image(tensor, filepath, normalize=True)


def make_hr_image(lr_tensor):
    # LR tensor をバイキュービックアップサンプルしてHR参照画像作成
    # lr_tensor: (B, 1, h, w)
    return torch.nn.functional.interpolate(lr_tensor, scale_factor=2, mode='bilinear', align_corners=False)


def make_lr_image(hr_tensor):
    # HR tensor をバイキュービックダウンしてLR参照画像作成
    return torch.nn.functional.interpolate(hr_tensor, scale_factor=0.5, mode='bilinear', align_corners=False)
