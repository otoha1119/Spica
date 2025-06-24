import argparse
import os
import pydicom
import torch
import numpy as np # NumPyをインポート
from sdflow.model_sdflow import SDFlowModel
from sdflow.utils_sdflow import get_device

def parse_args():
    """推論用の引数を解析します。"""
    parser = argparse.ArgumentParser(description="SDFlow Inference")
    parser.add_argument("--checkpoint", type=str, 
                        default="/workspace/checkpoints/sdflow_epoch31.pth",
                        help="使用する学習済みモデルのチェックポイントファイル (.pth)")
    parser.add_argument("--input", type=str,
                        default="/workspace/DataSet/ImageCAS/001.ImgCast/IM_091.dcm",
                        help="高解像度化したい低解像度DICOM画像のパス")
    # --- 出力のデフォルトを.dcmに変更 ---
    parser.add_argument("--output", type=str, default="result.dcm",
                        help="出力する高解像度DICOMファイル名")
    parser.add_argument("--scale", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()

    # 1. モデルのアーキテクチャを定義
    print("モデルを初期化しています...")
    # 注意: ここのパラメータは、読み込むチェックポイントを保存した時の設定と完全に一致させる必要があります
    model = SDFlowModel(
        in_channels=1, hidden_channels=64, n_flows=4, 
        hf_blocks=8, deg_blocks=4, deg_mixture=16, scale=args.scale
    ).to(device)

    # 2. 学習済みの重みをモデルに読み込む
    print(f"チェックポイント '{args.checkpoint}' を読み込んでいます...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("モデルの読み込みが完了しました。")

    # 3. 入力画像を読み込み、前処理を行う
    print(f"入力画像 '{args.input}' を読み込んでいます...")
    input_dicom = pydicom.dcmread(args.input)
    lr_image = input_dicom.pixel_array.astype('float32')

    # 学習時と全く同じ正規化処理
    lr_image_normalized = (lr_image + 1024) / 4095
    lr_tensor = torch.from_numpy(lr_image_normalized).unsqueeze(0).unsqueeze(0).to(device)
    lr_tensor = torch.clamp(lr_tensor, min=0, max=1)

    # 4. 推論を実行
    print("高解像度化処理を実行中...")
    with torch.no_grad():
        sr_tensor = model.generate_sr(lr_tensor, temp=0.0)
    print("高解像度化が完了しました。")

    # --- ここからが新しい処理 ---
    # 5. 結果を逆正規化し、DICOM形式に戻す
    print("結果をDICOM形式に変換しています...")
    
    # 5a. テンソルをCPUに戻し、NumPy配列に変換
    sr_numpy_normalized = sr_tensor.squeeze().cpu().numpy()
    
    # 5b. 正規化の逆の計算（逆正規化）を行って、元のピクセル値の範囲に戻す
    sr_numpy_denormalized = (sr_numpy_normalized * 4095.0) - 1024.0
    
    # 5c. DICOMで保存するために、適切な整数型に変換
    sr_final_pixels = sr_numpy_denormalized.astype(np.int16)

    # 6. 元のDICOM情報をテンプレートとして、ピクセルデータと一部のメタデータを更新
    output_dicom = input_dicom
    output_dicom.PixelData = sr_final_pixels.tobytes()
    output_dicom.Rows, output_dicom.Columns = sr_final_pixels.shape
    
    # 必要なDICOMタグがあればここで更新 (例)
    # output_dicom.SeriesDescription = "AI Super-Resolution"

    # 7. 新しいDICOMファイルとして保存
    output_dicom.save_as(args.output)
    print(f"結果をDICOMファイル '{args.output}' に保存しました。")


if __name__ == "__main__":
    main()