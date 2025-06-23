import argparse
import os
import pydicom
import torch
import numpy as np
import random
from torchvision.utils import save_image

# Datasetクラスをインポート
from sdflow.Dataset import DicomPairDataset

def main():
    parser = argparse.ArgumentParser(description="Dataset Test Script")
    parser.add_argument("--lr_root", type=str, default="/workspace/DataSet/ImageCAS", help="LR DICOM directory")
    parser.add_argument("--hr_root", type=str, default="/workspace/DataSet/photonCT/PhotonCT1024v2", help="HR DICOM directory")
    parser.add_argument("--output_dir", type=str, default="/workspace/dataset_test_output", help="Directory to save test images")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    args = parser.parse_args()

    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"テスト画像を '{args.output_dir}' に保存します。")

    # --- 1. 全体画像の処理 ---
    print("\n--- 1. 全体画像の正規化処理と統計値の確認 ---")
    
    # HRとLRの全ファイルパスを取得
    hr_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.hr_root) for name in files if name.lower().endswith('.dcm')]
    lr_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.lr_root) for name in files if name.lower().endswith('.dcm')]

    # HR全体画像を5枚処理
    print("\n[HR全体画像]")
    for i in range(args.num_samples):
        # ランダムにHR画像を1枚選択
        hr_path = random.choice(hr_files)
        hr_image = pydicom.dcmread(hr_path).pixel_array.astype(np.float32)
        
        # 最大値・最小値を出力
        print(f"  サンプル{i+1} ({os.path.basename(hr_path)}): 元の画素値 Min={hr_image.min():.1f}, Max={hr_image.max():.1f}")
        
        # Dataset.pyと同じ正規化を実行
        hr_normalized = (hr_image + 1024) / 4095
        hr_normalized_tensor = torch.from_numpy(hr_normalized).unsqueeze(0)
        hr_normalized_tensor = torch.clamp(hr_normalized_tensor, min=0, max=1)
        
        # 正規化後の画像をPNGとして保存（視覚的確認のため）
        save_image(hr_normalized_tensor, os.path.join(args.output_dir, f"hr_full_normalized_{i}.png"))

    # LR全体画像を5枚処理
    print("\n[LR全体画像]")
    for i in range(args.num_samples):
        lr_path = random.choice(lr_files)
        lr_image = pydicom.dcmread(lr_path).pixel_array.astype(np.float32)
        
        print(f"  サンプル{i+1} ({os.path.basename(lr_path)}): 元の画素値 Min={lr_image.min():.1f}, Max={lr_image.max():.1f}")
        
        lr_normalized = (lr_image + 1024) / 4095
        lr_normalized_tensor = torch.from_numpy(lr_normalized).unsqueeze(0)
        lr_normalized_tensor = torch.clamp(lr_normalized_tensor, min=0, max=1)

        save_image(lr_normalized_tensor, os.path.join(args.output_dir, f"lr_full_normalized_{i}.png"))
    
    print(f"\n> 正規化後の全体画像をPNGとして '{args.output_dir}' に保存しました。")


    # --- 2. クロップ画像の処理 ---
    print("\n--- 2. Dataset.pyからクロップ・正規化された画像の確認 ---")
    
    # Datasetを初期化
    dataset = DicomPairDataset(hr_root_dir=args.hr_root, lr_root_dir=args.lr_root, patch_size_hr=128, scale=2)
    
    # テンプレートDICOMを1つ読み込む（保存用）
    template_dicom = pydicom.dcmread(random.choice(hr_files))

    for i in range(args.num_samples):
        hr_patch_tensor, lr_patch_tensor, hr_full_image, lr_full_image = dataset[i]
        
        # 逆正規化して元の画素値の範囲に戻す
        hr_patch_denormalized = (hr_patch_tensor.squeeze().numpy() * 4095.0) - 1024.0
        lr_patch_denormalized = (lr_patch_tensor.squeeze().numpy() * 4095.0) - 1024.0
        
        # DICOMとして保存
        template_dicom.PixelData = hr_patch_denormalized.astype(np.int16).tobytes()
        template_dicom.Rows, template_dicom.Columns = hr_patch_denormalized.shape
        template_dicom.save_as(os.path.join(args.output_dir, f"hr_patch_{i}.dcm"))
        
        template_dicom.PixelData = lr_patch_denormalized.astype(np.int16).tobytes()
        template_dicom.Rows, template_dicom.Columns = lr_patch_denormalized.shape
        template_dicom.save_as(os.path.join(args.output_dir, f"lr_patch_{i}.dcm"))

    print(f"\n> Datasetから取得したクロップ画像をDICOMとして '{args.output_dir}' に保存しました。")
    print(f"\n--- 処理完了 ---")


if __name__ == "__main__":
    main()