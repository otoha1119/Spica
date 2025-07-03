import os
import argparse
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from sdflow.flow_block import FlowBlock
from sdflow.Dataset import DicomPairDataset
from sdflow.models import HRFlow, DegFlow
from sdflow.discriminators import ContentDiscriminator, ImageDiscriminator
from sdflow.loss_sdflow import LikelihoodLoss, LatentAdversarialLoss, PixelLoss, PerceptualLoss, ImageAdversarialLoss
from sdflow.utils_sdflow import get_device, save_checkpoint, save_visualizations

def parse_args():
    parser = argparse.ArgumentParser(description="SDFlow Training - Paper Aligned")
    parser.add_argument("--lr_root", type=str, default="/workspace/DataSet/ImageCAS", help="LR DICOM directory")
    parser.add_argument("--hr_root", type=str, default="/workspace/DataSet/photonCT/PhotonCT1024v2", help="HR DICOM directory")
    parser.add_argument("--output_dir", type=str, default="/workspace/checkpoints", help="Directory for checkpoints and logs")
    
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU (Paper: 32)")
    parser.add_argument("--patch_size_hr", type=int, default=192, help="Crop size for HR patches (Paper: 192)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for generator")
    parser.add_argument("--lr_disc", type=float, default=1e-5, help="Learning rate for discriminators")
    parser.add_argument("--scale", type=int, default=2, help="Super-resolution scale factor")
    parser.add_argument("--log_interval", type=int, default=100, help="Iterations between logging")
    
    # 3段階学習のための引数
    parser.add_argument("--stage1_iters", type=int, default=50000, help="Iterations for Stage 1 (NLL + L_content)")
    parser.add_argument("--stage2_iters", type=int, default=200000, help="Iterations for Stage 2 (Forward Loss)")
    
    parser.add_argument("--n_flows", type=int, default=4, help="Number of flow steps K in each FlowBlock (Paper: 16)")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels in Flow models")
    parser.add_argument("--n_gaussian", type=int, default=10, help="Number of gaussian mixtures for DegFlow")
    
    return parser.parse_args()

class LREncoder(nn.Module):
    """ LR画像を入力とし、z_c_lr と z_d を分離して出力するエンコーダ。 """
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.flow_block = FlowBlock(
            z_channels=in_channels, hidden_layers=3,
            hidden_channels=hidden_channels, n_steps=4,
            is_squeeze=True, is_split=False
        )
        self.out_conv = nn.Conv2d(in_channels * 4, out_channels * 2, 3, 1, 1)

    def forward(self, x):
        z, _ = self.flow_block(x, ldj=torch.zeros(x.size(0), device=x.device))
        z_combined = self.out_conv(z)
        z_c_lr, z_d = torch.chunk(z_combined, 2, dim=1)
        return z_c_lr, z_d

class ContentDecoder(nn.Module):
    """ 潜在変数z_cを入力とし、LR画像サイズの画像を再構成するデコーダ """
    def __init__(self, in_channels, out_channels=1, hidden_channels=64):
        super().__init__()
        self.main_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_channels, out_channels * 4, 3, 1, 1)
        )
        self.upsampler = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.main_net(x)
        x = self.upsampler(x)
        return x

def main():
    args = parse_args()
    device = get_device()

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    if os.path.exists(log_dir):
        print(f"古いログディレクトリ {log_dir} を削除しています...")
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    dataset = DicomPairDataset(hr_root_dir=args.hr_root, lr_root_dir=args.lr_root, patch_size_hr=args.patch_size_hr, scale=args.scale)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    in_channels = 1
    # models.pyのsplit_size=4に合わせる
    z_c_channels = (in_channels * 4 * 4) - 4
    
    hr_flow = HRFlow().to(device)
    deg_flow = DegFlow(in_channels=in_channels, cond_channels=z_c_channels, n_gaussian=args.n_gaussian).to(device)
    lr_encoder = LREncoder(in_channels=in_channels, out_channels=z_c_channels, hidden_channels=args.hidden_channels).to(device)
    content_decoder = ContentDecoder(in_channels=z_c_channels, out_channels=in_channels, hidden_channels=args.hidden_channels).to(device)

    flow_params = list(hr_flow.parameters()) + list(deg_flow.parameters()) + \
                  list(lr_encoder.parameters()) + list(content_decoder.parameters())

    disc_content = ContentDiscriminator(in_channels=z_c_channels).to(device)
    disc_hr = ImageDiscriminator(in_channels=in_channels).to(device)
    disc_lr = ImageDiscriminator(in_channels=in_channels).to(device)
    disc_params = list(disc_content.parameters()) + list(disc_hr.parameters()) + list(disc_lr.parameters())

    optim_flow = torch.optim.Adam(flow_params, lr=args.lr, betas=(0.9, 0.999))
    optim_disc = torch.optim.Adam(disc_params, lr=args.lr_disc, betas=(0.9, 0.999))
    
    total_steps = len(dataloader) * args.epochs
    milestones = [int(total_steps * 0.5), int(total_steps * 0.75), int(total_steps * 0.9), int(total_steps * 0.95)]
    scheduler_flow = torch.optim.lr_scheduler.MultiStepLR(optim_flow, milestones=milestones, gamma=0.5)
    scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(optim_disc, milestones=milestones, gamma=0.5)

    likelihood_loss = LikelihoodLoss().to(device)
    latent_adv_loss = LatentAdversarialLoss().to(device)
    pixel_loss = PixelLoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    image_adv_loss = ImageAdversarialLoss().to(device)

    # 3段階学習の境界を定義
    stage2_start_iter = args.stage1_iters
    stage3_start_iter = args.stage1_iters + args.stage2_iters

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        
        vis_lr_patch, vis_hr_patch, z_h_for_shape = None, None, None

        for idx, (hr_patch, lr_patch) in enumerate(progress_bar):
            
            if idx == 0:
                vis_hr_patch = hr_patch.to(device)
                vis_lr_patch = lr_patch.to(device)
                with torch.no_grad():
                    _, z_h_for_shape, _ = hr_flow(vis_hr_patch, ldj=torch.zeros(vis_hr_patch.size(0), device=device), reverse=False)

            hr_patch = hr_patch.to(device)
            lr_patch = lr_patch.to(device)

            # =========================
            #    ジェネレータの学習
            # =========================
            optim_flow.zero_grad()
            
            z_c_hr, z_h, logdet_y = hr_flow(hr_patch, ldj=torch.zeros(hr_patch.size(0), device=device), reverse=False)
            z_c_lr, z_d = lr_encoder(lr_patch)
            z_x, logdet_x_given_y = deg_flow(lr_patch, ldj=torch.zeros(hr_patch.size(0), device=device), u=z_c_hr.detach(), reverse=False)
            
            # --- 2. 損失の計算 ---
            loss_nll = likelihood_loss(z_h, logdet_y) + likelihood_loss(z_x, logdet_x_given_y)
            reconst_from_hr = content_decoder(z_c_hr); reconst_from_lr = content_decoder(z_c_lr)
            target_hr_downsampled = F.interpolate(hr_patch, scale_factor=1/args.scale, mode='bicubic', align_corners=False)
            target_lr = lr_patch
            lpf_reconst_from_hr = gaussian_blur(reconst_from_hr, kernel_size=5); lpf_reconst_from_lr = gaussian_blur(reconst_from_lr, kernel_size=5)
            lpf_target_for_hr = gaussian_blur(target_hr_downsampled, kernel_size=5); lpf_target_for_lr = gaussian_blur(target_lr, kernel_size=5)
            alpha = 0.05
            loss_content_l1_hr = pixel_loss(lpf_reconst_from_hr, lpf_target_for_hr); loss_content_l1_lr = pixel_loss(lpf_reconst_from_lr, lpf_target_for_lr)
            loss_content_per_hr = perceptual_loss(reconst_from_hr, target_hr_downsampled); loss_content_per_lr = perceptual_loss(reconst_from_lr, target_lr)
            loss_content = (loss_content_l1_hr + loss_content_l1_lr) + alpha * (loss_content_per_hr + loss_content_per_lr)
            
            # --- 変更点: loss_latent_gen を0で初期化 ---
            loss_latent_gen = torch.tensor(0.0, device=device)

            # --- ステージ1 損失 ---
            loss_G = 1.0 * loss_nll + 1.0 * loss_content

            # --- ステージ2 損失 ---
            if global_step >= stage2_start_iter:
                # 初期化したものを上書き
                loss_latent_gen = latent_adv_loss.generator_loss(disc_content, z_c_hr, z_c_lr)
                loss_G += 0.05 * loss_latent_gen
            
            # --- ステージ3 損失 ---
            sr_image, ds_image = None, None
            if global_step >= stage3_start_iter:
                sampled_h = torch.randn_like(z_h) * 0.8
                sr_input_z = [z_c_lr.detach(), sampled_h]
                sr_image, _ = hr_flow(sr_input_z, ldj=None, reverse=True)
                sampled_d = torch.randn_like(z_d)
                ds_image = content_decoder(z_c_hr.detach() + sampled_d)
                lambda_pix_ds=0.5; lambda_per_ds=0.5; lambda_gan_ds=0.1; lambda_pix_sr=0.5; lambda_per_sr=0.5; lambda_gan_sr=0.1
                loss_adv_hr = image_adv_loss.generator_loss(disc_hr, fake=sr_image); loss_adv_lr = image_adv_loss.generator_loss(disc_lr, fake=ds_image)
                loss_pix_sr = pixel_loss(sr_image, hr_patch); loss_per_sr = perceptual_loss(sr_image, hr_patch)
                loss_pix_ds = pixel_loss(ds_image, target_hr_downsampled); loss_per_ds = perceptual_loss(ds_image, target_hr_downsampled)
                loss_G += (lambda_pix_ds * loss_pix_ds + lambda_per_ds * loss_per_ds + lambda_gan_ds * loss_adv_lr) + \
                          (lambda_pix_sr * loss_pix_sr + lambda_per_sr * loss_per_sr + lambda_gan_sr * loss_adv_hr)

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(flow_params, 1.0)
            optim_flow.step()

            # === ディスクリミネータ学習 ===
            optim_disc.zero_grad()
            loss_D = torch.tensor(0.0, device=device)
            
            if global_step >= stage2_start_iter:
                z_LR = z_c_lr + z_d
                loss_dc = latent_adv_loss.disc_loss(disc_content, z_c_hr.detach(), z_c_lr.detach(), z_LR.detach())
                loss_D += loss_dc

            if global_step >= stage3_start_iter and sr_image is not None and ds_image is not None:
                loss_dhr = image_adv_loss.disc_loss(disc_hr, real=hr_patch, fake=sr_image.detach())
                loss_dlr = image_adv_loss.disc_loss(disc_lr, real=lr_patch, fake=ds_image.detach())
                loss_D += 0.5 * (loss_dhr + loss_dlr)
            
            if loss_D.item() > 0:
                loss_D.backward()
                optim_disc.step() 

            # === ログ記録 ===
            # /workspace/sdflow/train_sdflow.py の main 関数内

            # --- ログ記録 ---
            if global_step % args.log_interval == 0:
                writer.add_scalar("Loss/Gen_Total", loss_G.item(), global_step)
                writer.add_scalar("Loss/Disc_Total", loss_D.item(), global_step)
                writer.add_scalar("Loss_Comp/NLL", loss_nll.item(), global_step)
                writer.add_scalar("Loss_Comp/Content", loss_content.item(), global_step)
                if 'loss_latent_gen' in locals(): # loss_latent_genが存在する場合のみログを記録
                    writer.add_scalar("Loss_Comp/Latent_Adv", loss_latent_gen.item(), global_step)
                
                with torch.no_grad():
                    # --- 変更点: 変数名のタイポを修正 (zz_c_lr_vis -> z_c_lr_vis) ---
                    z_c_lr_vis, _ = lr_encoder(lr_patch) # z_dは使わないので _ で捨てる
                    vis_sampled_h = torch.randn_like(z_h)
                    
                    vis_sr_input = [z_c_lr_vis, vis_sampled_h]
                    sr_vis, _ = hr_flow(vis_sr_input, ldj=None, reverse=True)
                    # --- 変更ここまで ---
                    
                    writer.add_image("Images/A_Input_LR", lr_patch[0], global_step)
                    writer.add_image("Images/B_Output_SR", sr_vis[0], global_step)
                    writer.add_image("Images/C_GroundTruth_HR", hr_patch[0], global_step)
            
            global_step += 1
        
        scheduler_flow.step()
        scheduler_disc.step()
        
        # === エポック終了時の可視化と保存 ===
        with torch.no_grad():
            if vis_lr_patch is not None:
                z_c_lr_vis, _ = lr_encoder(vis_lr_patch)
                vis_sampled_h = torch.randn_like(z_h_for_shape)
                vis_sr_input = [z_c_lr_vis, vis_sampled_h]
                sr_vis, _ = hr_flow(vis_sr_input, ldj=None, reverse=True)
                save_visualizations(vis_lr_patch[0], sr_vis[0].detach(), vis_hr_patch[0], args.output_dir, epoch)
            
        models_to_save = {
            'hr_flow': hr_flow, 'deg_flow': deg_flow, 'lr_encoder': lr_encoder, 
            'content_decoder': content_decoder, 'disc_content': disc_content,
            'disc_hr': disc_hr, 'disc_lr': disc_lr
        }
        optims_to_save = {'flow': optim_flow, 'disc': optim_disc}
        checkpoint_path = os.path.join(args.output_dir, "checkpoints", f"epoch_{epoch}.pth")
        save_checkpoint(models_to_save, optims_to_save, epoch, checkpoint_path)
        print(f"\n[Epoch {epoch}] Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()