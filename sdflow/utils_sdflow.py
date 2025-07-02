# /workspace/sdflow/utils_sdflow.py

import os
import torch
import torchvision.utils as vutils
from torch.nn.functional import interpolate

def get_device():
    """
    CUDA が使えれば CUDA、そうでなければ CPU を返す
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 変更点: save_checkpoint と load_checkpoint を全面的に修正 ---

def save_checkpoint(models_dict, optims_dict, epoch, path):
    """
    複数のモデルとオプティマイザの状態をまとめてファイルに保存する

    Parameters:
    - models_dict: {'モデル名': model_instance, ...} の辞書
    - optims_dict: {'オプティマイザ名': optim_instance, ...} の辞書
    - epoch: 現在のエポック数
    - path: チェックポイントファイルの保存パス
    """
    # ディレクトリが存在しなければ作成
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存するデータを準備
    state_to_save = {'epoch': epoch}
    
    # 各モデルのstate_dictを保存
    for name, model in models_dict.items():
        state_to_save[name + '_state'] = model.state_dict()
        
    # 各オプティマイザのstate_dictを保存
    for name, optim in optims_dict.items():
        state_to_save[name + '_state'] = optim.state_dict()

    torch.save(state_to_save, path)


def load_checkpoint(models_dict, optims_dict, path):
    """
    ファイルからモデルとオプティマイザの状態を読み込む

    Parameters:
    - models_dict: {'モデル名': model_instance, ...} の辞書
    - optims_dict: {'オプティマイザ名': optim_instance, ...} の辞書
    - path: チェックポイントファイルのパス

    Returns:
    - epoch: 保存されていたエポック数
    """
    if not os.path.exists(path):
        print(f"警告: チェックポイントファイルが見つかりません: {path}")
        return 0

    checkpoint = torch.load(path)
    
    # 各モデルにstate_dictを読み込ませる
    for name, model in models_dict.items():
        key = name + '_state'
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
            print(f"モデル '{name}' の重みを読み込みました。")
        else:
            print(f"警告: チェックポイントにモデル '{name}' の重みが見つかりません。")

    # 各オプティマイザにstate_dictを読み込ませる
    for name, optim in optims_dict.items():
        key = name + '_state'
        if key in checkpoint:
            optim.load_state_dict(checkpoint[key])
            print(f"オプティマイザ '{name}' の状態を読み込みました。")
        else:
            print(f"警告: チェックポイントにオプティマイザ '{name}' の状態が見つかりません。")

    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"エポック {start_epoch} から学習を再開します。")
    
    return start_epoch

# --- 変更ここまで ---

def save_visualizations(lr, sr, hr, output_dir, epoch):
    """
    学習中、エポックごとに可視化用ディレクトリを作成し、
    LR入力、生成されたSR、参照HRを画像として保存する
    """
    vis_dir = os.path.join(output_dir, "vis", f"epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)
    # 正規化して PNG 保存
    vutils.save_image(lr, os.path.join(vis_dir, "lr.png"), normalize=True)
    vutils.save_image(sr, os.path.join(vis_dir, "sr.png"), normalize=True)
    vutils.save_image(hr, os.path.join(vis_dir, "hr.png"), normalize=True)


def make_bicubic_hr(lr, scale):
    """
    LR → HR のバイキュービックスケーリング参照画像を生成

    Parameters:
    - lr: LR画像テンソル（[B,1,L,L]）
    - scale: スケールファクタ (例: 4)

    Returns:
    - バイキュービック補間された HR 画像テンソル（[B,1, L*scale, L*scale]）
    """
    return interpolate(lr, scale_factor=scale, mode='bilinear', align_corners=False)


def make_bicubic_lr(hr, scale):
    """
    HR → LR のバイキュービックスケーリング参照画像を生成

    Parameters:
    - hr: HR画像テンソル（[B,1,Hr,Hr]）
    - scale: スケールファクタ (例: 4)

    Returns:
    - バイキュービック補間された LR 画像テンソル（[B,1, Hr/scale, Hr/scale]）
    """
    return interpolate(hr, scale_factor=1/scale, mode='bilinear', align_corners=False)
