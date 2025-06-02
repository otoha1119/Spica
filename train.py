import os
import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Dataset import DicomPairDataset
from model import SDFlowModel, HRDiscriminator, LRDiscriminator, LatentDiscriminator
from loss import (
    LikelihoodLoss,
    ContentLoss,
    AdversarialLoss,
    PixelLoss,
    PerceptualLoss
)
from utils import (
    save_image_tensor,
    get_device,
    make_lr_image,
    make_hr_image
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SDFlow for unpaired super-resolution")
    parser.add_argument("--lr_root", type=str, default="/workspace/DataSet/ImageCAS", help="Path to LR DICOM directory")
    parser.add_argument("--hr_root", type=str, default="/workspace/DataSet/photonCT/PhotonCT1024", help="Path to HR DICOM directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints and logs")
    parser.add_argument("--log_interval", type=int, default=100, help="Iterations between logging to TensorBoard")
    parser.add_argument("--save_interval", type=int, default=1, help="Epoch interval to save checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # データセットとデータローダー
    train_dataset = DicomPairDataset(args.hr_root, args.lr_root)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # モデルと判別器の初期化
    sdflow = SDFlowModel().to(device)
    d_hr = HRDiscriminator().to(device)
    d_lr = LRDiscriminator().to(device)
    d_latent = LatentDiscriminator().to(device)

    # オプティマイザ
    optimizer_G = torch.optim.Adam(sdflow.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(
        list(d_hr.parameters()) + list(d_lr.parameters()) + list(d_latent.parameters()),
        lr=args.lr, betas=(0.9, 0.999)
    )

    # 学習率スケジューラ
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[args.epochs//4, args.epochs//2, args.epochs*3//4], gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[args.epochs//4, args.epochs//2, args.epochs*3//4], gamma=0.5)

    # 損失関数
    likelihood_loss = LikelihoodLoss().to(device)
    content_loss = ContentLoss().to(device)
    adv_loss = AdversarialLoss().to(device)
    pixel_loss = PixelLoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)

    # TensorBoard ログ
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        sdflow.train()
        d_hr.train()
        d_lr.train()
        d_latent.train()

        for i, (hr_tensor, lr_tensor) in enumerate(train_loader):
            hr_tensor = hr_tensor.to(device)
            lr_tensor = lr_tensor.to(device)

            # -------------------------
            #  (1) Generator (SDFlow) の訓練
            # -------------------------
            optimizer_G.zero_grad()

            # 正方向: HR->(z_c, z_h)、LR->(z_c, z_d)
            z_c_hr, z_h, z_c_lr_from_hr, z_d_from_hr, logdet_hr = sdflow.forward_hr(hr_tensor)
            z_c_lr, z_d, z_c_hr_from_lr, z_h_from_lr, logdet_lr = sdflow.forward_lr(lr_tensor)

            # 対数尤度損失
            loss_nll_hr = likelihood_loss.hr_loss(z_h, logdet_hr)
            loss_nll_lr = likelihood_loss.lr_loss(z_d, logdet_lr)

            # 潜在空間GAN損失
            loss_latent_G = adv_loss.generator_loss_latent(d_latent, z_c_hr.detach(), z_c_lr.detach())

            # 逆方向: サンプリング手法（生成画像）
            # 温度0: 平均像
            sr_mean = sdflow.generate_sr(lr_tensor, temp=0)
            ds_mean = sdflow.generate_ds(hr_tensor, temp=0)
            # 温度0.8: 多様性サンプル
            sr_rand = sdflow.generate_sr(lr_tensor, temp=0.8)
            ds_rand = sdflow.generate_ds(hr_tensor, temp=0.8)

            # コンテンツ一貫性損失
            loss_content = content_loss(hr_tensor, ds_mean, z_c_hr, sdflow)

            # 生成画像の各種損失 (SRとDS)
            loss_sr_pix = pixel_loss(sr_mean, make_hr_image(lr_tensor))
            loss_sr_per = perceptual_loss(sr_mean, make_hr_image(lr_tensor))
            loss_sr_g = adv_loss.generator_loss_image(d_hr, sr_rand)

            loss_ds_pix = pixel_loss(ds_mean, make_lr_image(hr_tensor))
            loss_ds_per = perceptual_loss(ds_mean, make_lr_image(hr_tensor))
            loss_ds_g = adv_loss.generator_loss_image(d_lr, ds_rand)

            # 合計Generator損失
            loss_G = (
                loss_nll_hr + loss_nll_lr
                + 0.05 * loss_content
                + 0.05 * loss_latent_G
                + 0.5 * loss_sr_pix + 0.5 * loss_sr_per + 0.1 * loss_sr_g
                + 0.5 * loss_ds_pix + 0.5 * loss_ds_per + 0.1 * loss_ds_g
            )
            loss_G.backward()
            optimizer_G.step()

            # -------------------------
            #  (2) Discriminator の訓練
            # -------------------------
            optimizer_D.zero_grad()

            # 画像判別器損失
            loss_d_hr = adv_loss.discriminator_loss_image(d_hr, make_hr_image(lr_tensor).detach(), sr_rand.detach())
            loss_d_lr = adv_loss.discriminator_loss_image(d_lr, make_lr_image(hr_tensor).detach(), ds_rand.detach())

            # 潜在判別器損失
            loss_d_latent = adv_loss.discriminator_loss_latent(d_latent, z_c_hr.detach(), z_c_lr.detach())

            loss_D = loss_d_hr + loss_d_lr + 0.5 * loss_d_latent
            loss_D.backward()
            optimizer_D.step()

            # -------------------------
            #  (3) ログ出力
            # -------------------------
            if global_step % args.log_interval == 0:
                writer.add_scalar("Loss/G_Total", loss_G.item(), global_step)
                writer.add_scalar("Loss/D_Total", loss_D.item(), global_step)
                writer.add_scalar("Loss/NLL_HR", loss_nll_hr.item(), global_step)
                writer.add_scalar("Loss/NLL_LR", loss_nll_lr.item(), global_step)

                # サンプル画像をTensorBoardに記録
                writer.add_image("LR_Input", lr_tensor[0], global_step)
                writer.add_image("SR_Output", sr_mean[0].detach(), global_step)
                writer.add_image("HR_Reference", make_hr_image(lr_tensor)[0], global_step)

            global_step += 1

        # エポック終了後のスケジューラ更新
        scheduler_G.step()
        scheduler_D.step()

        # -------------------------
        #  モデル保存
        # -------------------------
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"sdflow_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': sdflow.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict()
            }, ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

    writer.close()

if __name__ == '__main__':
    main()
