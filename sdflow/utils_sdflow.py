import os
import torch
import torchvision.utils as vutils
from torch.nn.functional import interpolate


def get_device():
    """
    CUDA が使えれば CUDA、そうでなければ CPU を返す
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, disc_content, disc_hr, disc_lr,
                    optim_flow, optim_disc, epoch, path):
    """
    モデルと識別器、オプティマイザの状態をまとめてファイルに保存する

    Parameters:
    - model: SDFlowModel インスタンス
    - disc_content: ContentDiscriminator インスタンス
    - disc_hr: ImageDiscriminator (HR側) インスタンス
    - disc_lr: ImageDiscriminator (LR側) インスタンス
    - optim_flow: Flowモデル用オプティマイザ
    - optim_disc: 識別器用オプティマイザ
    - epoch: 現在のエポック数
    - path: チェックポイントファイルの保存パス
    """
    # ディレクトリが存在しなければ作成
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'disc_content_state': disc_content.state_dict(),
        'disc_hr_state': disc_hr.state_dict(),
        'disc_lr_state': disc_lr.state_dict(),
        'optim_flow_state': optim_flow.state_dict(),
        'optim_disc_state': optim_disc.state_dict()
    }, path)


def save_visualizations(lr, sr, hr, output_dir, epoch):
    """
    学習中、エポックごとに可視化用ディレクトリを作成し、
    LR入力、生成されたSR、参照HRを画像として保存する

    Parameters:
    - lr: LR画像テンソル（[1,H,W] または [1,L,L]）
    - sr: 生成されたSR画像テンソル（[1,Hr,Hr]）
    - hr: 参照HR画像テンソル（[1,Hr,Hr]）
    - output_dir: チェックポイント／可視化出力のベースディレクトリ
    - epoch: 現在のエポック数
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
