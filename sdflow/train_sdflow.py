###############################################
# sdflow/train_sdflow.py
###############################################
import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import make_grid     
import torch.nn.functional as F 

from sdflow.Dataset import DicomPairDataset
from sdflow.model_sdflow import SDFlowModel
from sdflow.discriminators import ContentDiscriminator, ImageDiscriminator
from sdflow.loss_sdflow import (
    LikelihoodLoss, LatentAdversarialLoss,
    PixelLoss, PerceptualLoss, ImageAdversarialLoss
)
from sdflow.utils_sdflow import (
    get_device, save_checkpoint, save_visualizations,
    make_bicubic_lr, make_bicubic_hr
)


def parse_args():
    parser = argparse.ArgumentParser(description="SDFlow Training")
    parser.add_argument("--lr_root", type=str,
                        default="/workspace/DataSet/ImageCAS", help="LR DICOM directory")
    parser.add_argument("--hr_root", type=str,
                        default="/workspace/DataSet/photonCT/PhotonCT1024", help="HR DICOM directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for flows and discriminators")
    parser.add_argument("--output_dir", type=str, default="/workspace/checkpoints", help="Directory for checkpoints and logs")
    parser.add_argument("--patch_size_hr", type=int, default=128, help="Crop size for HR patches")
    parser.add_argument("--scale", type=int, default=2, help="Super-resolution scale factor")
    parser.add_argument("--log_interval", type=int, default=100, help="Iterations between logging")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Dataset and DataLoader (unpaired)
    dataset = DicomPairDataset(hr_root_dir=args.hr_root,
                               lr_root_dir=args.lr_root,
                               patch_size_hr=args.patch_size_hr,
                               scale=args.scale)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=8, pin_memory=True)

    # Models
    model = SDFlowModel(in_channels=1, hidden_channels=64,
                        n_levels=3, n_flows=4,
                        hf_blocks=8, deg_blocks=4,
                        deg_mixture=16,
                        scale=args.scale).to(device)

    # Discriminators
    #disc_content = ContentDiscriminator(in_channels=32).to(device)
    disc_content = ContentDiscriminator(in_channels=1).to(device)
    disc_hr = ImageDiscriminator(in_channels=1).to(device)
    disc_lr = ImageDiscriminator(in_channels=1).to(device)

    # Optimizers
    optim_flow = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optim_disc = torch.optim.Adam(
        list(disc_content.parameters()) +
        list(disc_hr.parameters()) +
        list(disc_lr.parameters()),
        lr=args.lr, betas=(0.9, 0.999)
    )

    # Learning rate schedulers
    total_steps = len(dataloader) * args.epochs
    milestones = [total_steps // 4, total_steps // 2, total_steps * 3 // 4]
    scheduler_flow = torch.optim.lr_scheduler.MultiStepLR(optim_flow,
                                                          milestones=milestones, gamma=0.5)
    scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(optim_disc,
                                                          milestones=milestones, gamma=0.5)

    # Loss functions
    likelihood_loss = LikelihoodLoss().to(device)
    latent_adv_loss = LatentAdversarialLoss().to(device)
    pixel_loss = PixelLoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    image_adv_loss = ImageAdversarialLoss().to(device)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        disc_content.train()
        disc_hr.train()
        disc_lr.train()

        # tqdmでデータローダーをラップし、エポック情報を表示
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)

        for idx, (hr_patch, lr_patch) in enumerate(progress_bar):
            # ループの最初に、このステップで表示する損失値を初期化
            loss_G_val = 0.0
            loss_D_val = 0.0

            hr_patch = hr_patch.to(device)  # [B,1,Hr,Hr]
            lr_patch = lr_patch.to(device)  # [B,1,Lr,Lr]

            # -------------------------
            # Generator (Flow) の学習
            # -------------------------
            optim_flow.zero_grad()
            
            # Forward flow: encode HR and LR
            z_c_hr, z_h, logdet_h = model.encode_hr(hr_patch)
            z_c_lr, z_d, logdet_d = model.encode_lr(lr_patch)

            # Negative log-likelihood losses
            loss_nll_h = likelihood_loss(z_h, logdet_h)
            loss_nll_d = likelihood_loss(z_d, logdet_d)
            
            # Latent adversarial loss (for generator)
            loss_latent_gen  = latent_adv_loss.generator_loss(disc_content, z_c_hr, z_c_lr)

            # Backward generation (sampling outputs for adversarial loss)
            sr_rand = model.generate_sr(lr_patch, temp=0.8)
            ds_rand = model.generate_ds(hr_patch, temp=0.8)

            # Image adversarial losses (for generator)
            loss_adv_hr = image_adv_loss.generator_loss(disc_hr, fake=sr_rand)
            loss_adv_lr = image_adv_loss.generator_loss(disc_lr, fake=ds_rand)

            # Per-pixel and perceptual losses (use mean outputs for stability)
            with torch.no_grad(): # この部分は勾配計算不要
                sr_mean = model.generate_sr(lr_patch, temp=0.0)
                ds_mean = model.generate_ds(hr_patch, temp=0.0)
            bicubic_hr = make_bicubic_hr(lr_patch, scale=args.scale)
            bicubic_lr = make_bicubic_lr(hr_patch, scale=args.scale)
            loss_pix_sr = pixel_loss(sr_mean, bicubic_hr)
            loss_per_sr = perceptual_loss(sr_mean, bicubic_hr)
            loss_pix_ds = pixel_loss(ds_mean, bicubic_lr)
            loss_per_ds = perceptual_loss(ds_mean, bicubic_lr)

            # Total generator (flow) loss
            loss_G = (
                loss_nll_h + loss_nll_d
                + 0.05 * loss_latent_gen
                + 0.5 * (loss_pix_sr + loss_per_sr + loss_pix_ds + loss_per_ds)
                + 0.1 * (loss_adv_hr + loss_adv_lr)
            )
          
            loss_G.backward()
            optim_flow.step()
            loss_G_val = loss_G.item() # 表示用に損失値を保存

            torch.cuda.empty_cache()

            # -------------------------
            # Discriminator の学習
            # -------------------------
            optim_disc.zero_grad()
            
            # Content discriminator loss (real HR vs LR)
            loss_dc = latent_adv_loss.disc_loss(disc_content, z_c_hr.detach(), z_c_lr.detach())
            
            # Image discriminator losses
            loss_dhr = image_adv_loss.disc_loss(disc_hr, real=hr_patch, fake=sr_rand.detach())
            loss_dlr = image_adv_loss.disc_loss(disc_lr, real=lr_patch, fake=ds_rand.detach())
            
            loss_D = loss_dc + 0.5 * (loss_dhr + loss_dlr)
            
            loss_D.backward()
            optim_disc.step()
            loss_D_val = loss_D.item() # 表示用に損失値を保存

            # プログレスバーに最新の損失値を表示
            progress_bar.set_postfix(loss_G=loss_G_val, loss_D=loss_D_val)

            # Logging
            if global_step % args.log_interval == 0:    
                lr_resized = F.interpolate(lr_patch, size=sr_mean.shape[-2:], mode='bicubic', align_corners=False)
                # 2枚の画像を横に並べたグリッド画像を作成
                image_grid = make_grid([lr_resized[0], sr_mean[0].detach()], nrow=2, normalize=True)
                writer.add_image("Comparison/Input_LR_and_Output_SR", image_grid, global_step)
                writer.add_image("HR_GroundTruth", hr_patch[0], global_step)
                
                writer.add_scalar("Loss/Gen_Total", loss_G.item(), global_step)
                writer.add_scalar("Loss/Disc_Total", loss_D.item(), global_step)
                writer.add_scalar("Loss/NLL_H", loss_nll_h.item(), global_step)
                writer.add_scalar("Loss/NLL_D", loss_nll_d.item(), global_step)      

            global_step += 1

        # --- end of batch loop ---

        # Scheduler step per epoch
        scheduler_flow.step()
        scheduler_disc.step()

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(args.output_dir, f"sdflow_epoch{epoch}.pth")
        save_checkpoint(model, disc_content, disc_hr, disc_lr,
                        optim_flow, optim_disc, epoch, checkpoint_path)
        print(f"\n[Epoch {epoch}] Checkpoint saved to {checkpoint_path}")

        # Save example visualizations
        with torch.no_grad():
            # 1番目のデータで可視化サンプルを生成
            vis_lr = dataset[0][1].unsqueeze(0).to(device)
            vis_hr = dataset[0][0].unsqueeze(0).to(device)
            vis_sr = model.generate_sr(vis_lr, temp=0.0)
            save_visualizations(vis_lr[0], vis_sr[0].detach(), vis_hr[0],
                                args.output_dir, epoch)

    writer.close()


if __name__ == "__main__":
    main()