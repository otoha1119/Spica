import argparse
import os
import pydicom
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# 必要なモデル定義をインポート
from sdflow.model_sdflow import SDFlowModel
from sdflow.utils_sdflow import get_device

def parse_args():
    """推論用の引数を解析します。"""
    parser = argparse.ArgumentParser(description="SDFlow Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="使用する学習済みモデルのチェックポイントファイル (.pth)")
    parser.add_argument("--input", type=str, required=True,
                        help="高解像度化したい低解像度DICOM画像のパス")
    parser.add_argument("--output", type=str, default="result.png",
                        help="出力する高解像度画像ファイル名")
    parser.add_argument("--scale", type=int, default=2,
                        help="超解像のスケール（学習時と同じ値を指定）")
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()

    # 1. モデルのアーキテクチャを定義し、デバイスに送る
    # 注意: ここのパラメータは、学習時に使ったものと一致させる必要があります
    print("モデルを初期化しています...")
    model = SDFlowModel(
        in_channels=1,
        hidden_channels=32,  # エラーから判断すると、おそらく32
        n_flows=3,           # ← 学習時と同じ値に修正してください
        hf_blocks=4,         # ← 学習時と同じ値に修正してください
        deg_blocks=2,        # ← 学習時と同じ値に修正してください
        deg_mixture=16,      # エラーから判断すると、おそらく16
        scale=args.scale
    ).to(device)

    # 2. 学習済みの重みをモデルに読み込む
    print(f"チェックポイント '{args.checkpoint}' を読み込んでいます...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()  # 推論モードに設定（非常に重要）
    print("モデルの読み込みが完了しました。")

    # 3. 入力画像を読み込み、前処理を行う
    print(f"入力画像 '{args.input}' を読み込んでいます...")
    hr_dicom = pydicom.dcmread(args.input)
    hr_image = hr_dicom.pixel_array.astype('float32')

    # 学習時と同じ正規化を適用
    hr_image_normalized = (hr_image + 2048.0) / 6143.0
    
    # PyTorchテンソルに変換し、バッチ次元とチャンネル次元を追加 [H, W] -> [1, 1, H, W]
    lr_tensor = torch.from_numpy(hr_image_normalized).unsqueeze(0).unsqueeze(0).to(device)
    lr_tensor = torch.clamp(lr_tensor, min=0) # 念のためクリッピング

    # 4. 推論を実行
    print("高解像度化処理を実行中...")
    with torch.no_grad():  # 勾配計算を無効化し、メモリ効率と速度を向上
        # temp=0.0 で決定論的な（毎回同じ）出力を得る
        sr_tensor = model.generate_sr(lr_tensor, temp=0.0)
    print("高解像度化が完了しました。")

    # 5. 結果を画像として保存
    # テンソルの値が0-1の範囲なので、そのまま保存できます
    save_image(sr_tensor, args.output, normalize=False)
    print(f"結果を '{args.output}' に保存しました。")


if __name__ == "__main__":
    main()