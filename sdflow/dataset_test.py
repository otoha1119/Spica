import argparse
import os
import pydicom
import torch
import numpy as np
import random
from torchvision.utils import save_image

# Datasetクラスをインポート
from sdflow.Dataset import DicomPairDataset

def save_numpy_as_dicom(pixel_array, template_dicom, output_path, series_description="Test Output"):
    """NumPy配列をDICOMファイルとして保存するヘルパー関数"""
    new_dicom = template_dicom.copy()
    new_dicom.PixelData = pixel_array.astype(np.int16).tobytes()
    new_dicom.Rows, new_dicom.Columns = pixel_array.shape
    new_dicom.SeriesDescription = series_description
    new_dicom.save_as(output_path)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Dataset Test Script")
    parser.add_argument("--lr_root", type=str, default="/workspace/DataSet/ImageCAS", help="LR DICOM directory")
    parser.add_argument("--hr_root", type=str, default="/workspace/DataSet/photonCT/PhotonCT1024v2", help="HR DICOM directory")
    parser.add_argument("--output_dir", type=str, default="/workspace/dataset_test_output", help="Directory to save test images")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"テスト画像を '{args.output_dir}' に保存します。")

    dataset = DicomPairDataset(
        hr_root_dir=args.hr_root,
        lr_root_dir=args.lr_root,
        patch_size_hr=128,
        scale=2
    )
    
    any_hr_file_path = os.path.join(dataset.hr_dirs[0], os.listdir(dataset.hr_dirs[0])[0])
    dicom_template = pydicom.dcmread(any_hr_file_path)

    print(f"\n--- {args.num_samples}個のサンプルを処理して検証します ---")
    for i in range(args.num_samples):
        print(f"\n=============== サンプル {i+1} ================")
        
        hr_image, lr_image, hr_crop, lr_crop, hr_tensor, lr_tensor = dataset[i]

        # 1. 全体画像
        print("[1. 全体画像]")
        print(f"  HR Full | 元の画素値 Min={hr_image.min():.1f}, Max={hr_image.max():.1f}")
        print(f"  LR Full | 元の画素値 Min={lr_image.min():.1f}, Max={lr_image.max():.1f}")
        save_numpy_as_dicom(hr_image, dicom_template, os.path.join(args.output_dir, f"sample_{i}_1_hr_full.dcm"), "Full HR Original")
        save_numpy_as_dicom(lr_image, dicom_template, os.path.join(args.output_dir, f"sample_{i}_1_lr_full.dcm"), "Full LR Original")
        print("  > DICOMとして保存しました。")

        # 2. 全体画像を正規化したもの
        print("\n[2. 全体画像（正規化後）]")
        hr_full_normalized = (hr_image + 1024) / 4095
        lr_full_normalized = (lr_image + 1024) / 4095
        print(f"  HR Full Normalized | Min={hr_full_normalized.min():.4f}, Max={hr_full_normalized.max():.4f}")
        print(f"  LR Full Normalized | Min={lr_full_normalized.min():.4f}, Max={lr_full_normalized.max():.4f}")
        save_image(torch.from_numpy(hr_full_normalized).unsqueeze(0), os.path.join(args.output_dir, f"sample_{i}_2_hr_full_normalized.png"))
        save_image(torch.from_numpy(lr_full_normalized).unsqueeze(0), os.path.join(args.output_dir, f"sample_{i}_2_lr_full_normalized.png"))
        print("  > PNGとして保存しました。")

        # 3. クロップした画像
        print("\n[3. クロップ画像（正規化前）]")
        print(f"  HR Crop | 元の画素値 Min={hr_crop.min():.1f}, Max={hr_crop.max():.1f}")
        print(f"  LR Crop | 元の画素値 Min={lr_crop.min():.1f}, Max={lr_crop.max():.1f}")
        save_numpy_as_dicom(hr_crop, dicom_template, os.path.join(args.output_dir, f"sample_{i}_3_hr_crop_original.dcm"), "Cropped HR (Original)")
        save_numpy_as_dicom(lr_crop, dicom_template, os.path.join(args.output_dir, f"sample_{i}_3_lr_crop_original.dcm"), "Cropped LR (Original)")
        print("  > DICOMとして保存しました。")

        # 4. クロップして正規化した画像
        print("\n[4. クロップ画像（正規化後）]")
        print(f"  HR Patch Normalized | Min={hr_tensor.min():.4f}, Max={hr_tensor.max():.4f}")
        print(f"  LR Patch Normalized | Min={lr_tensor.min():.4f}, Max={lr_tensor.max():.4f}")
        save_image(hr_tensor, os.path.join(args.output_dir, f"sample_{i}_4_hr_patch_normalized.png"))
        save_image(lr_tensor, os.path.join(args.output_dir, f"sample_{i}_4_lr_patch_normalized.png"))
        print("  > PNGとして保存しました。")

        # 5. クロップして逆正規化した画像
        print("\n[5. クロップ画像（逆正規化後）]")
        hr_patch_denormalized = (hr_tensor.squeeze().numpy() * 4095.0) - 1024.0
        lr_patch_denormalized = (lr_tensor.squeeze().numpy() * 4095.0) - 1024.0
        print(f"  HR Patch De-Normalized | Min={hr_patch_denormalized.min():.1f}, Max={hr_patch_denormalized.max():.1f}")
        print(f"  LR Patch De-Normalized | Min={lr_patch_denormalized.min():.1f}, Max={lr_patch_denormalized.max():.1f}")
        save_numpy_as_dicom(hr_patch_denormalized, dicom_template, os.path.join(args.output_dir, f"sample_{i}_5_hr_patch_final.dcm"), "Cropped HR (Final)")
        save_numpy_as_dicom(lr_patch_denormalized, dicom_template, os.path.join(args.output_dir, f"sample_{i}_5_lr_patch_final.dcm"), "Cropped LR (Final)")
        print("  > DICOMとして保存しました。")

    print(f"\n--- 処理完了 ---")

if __name__ == "__main__":
    main()