import os
import argparse
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn # シンプルなエンコーダ定義のために追加
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# --- 変更点: これまで修正したモジュールを正しくインポート ---
from sdflow.flow_block import FlowBlock
from sdflow.Dataset import DicomPairDataset
from sdflow.models import HRFlow, ContentFlow, DegFlow
from sdflow.discriminators import ContentDiscriminator, ImageDiscriminator
from sdflow.loss_sdflow import LikelihoodLoss, LatentAdversarialLoss, PixelLoss, PerceptualLoss, ImageAdversarialLoss
from torchvision.transforms.functional import gaussian_blur
from sdflow.utils_sdflow import get_device, save_checkpoint, save_visualizations
# --- 変更ここまで ---

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
    parser.add_argument("--pretrain_steps", type=int, default=50000, help="Number of pre-training iterations (Paper: 50k)")
    
    # --- 変更点なし: パラメータはそのまま使用 ---
    parser.add_argument("--n_flows", type=int, default=4, help="Number of flow steps K in each FlowBlock (Paper: 16)")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels in Flow models")
    parser.add_argument("--n_gaussian", type=int, default=10, help="Number of gaussian mixtures for DegFlow")
    
    return parser.parse_args()

# --- 変更点: LatentAdversarialLossで比較するため、LR画像からz_cを生成する簡易エンコーダを定義 ---
class LREncoder(nn.Module):
    """
    LR画像を入力とし、HRFlowの潜在空間と同じサイズのz_c_lrを出力するエンコーダ。
    """
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        # LR画像(96x96)を1回Squeezeして、潜在空間(48x48)に合わせる
        self.flow_block = FlowBlock(
            z_channels=in_channels,
            hidden_layers=3,
            hidden_channels=hidden_channels,
            n_steps=4,
            is_squeeze=True, # 空間サイズを半分に、チャネルを4倍に
            is_split=False   # 分割はしない
        )
        # Squeeze後のチャネル数(1*4=4)を、目的のz_c_channelsに変換する
        self.out_conv = nn.Conv2d(in_channels * 4, out_channels, 3, 1, 1)

    def forward(self, x):
        # ldjは使わないので _ で受ける
        z, _ = self.flow_block(x, ldj=torch.zeros(x.size(0), device=x.device))
        return self.out_conv(z)
# --- 変更ここまで ---

class ContentDecoder(nn.Module):
    """
    潜在変数z_c(48x48)を入力とし、LR画像サイズ(96x96)の画像を再構成するデコーダ
    """
    def __init__(self, in_channels, out_channels=1, hidden_channels=64):
        super().__init__()
        # --- 変更点: PixelShuffle(x2)のため、チャンネル数を 4 * out_channels にする ---
        self.main_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_channels, out_channels * 4, 3, 1, 1) # 16から4に変更
        )
        self.upsampler = nn.PixelShuffle(2) # 4から2に変更

    def forward(self, x):
        x = self.main_net(x)
        x = self.upsampler(x)
        return x

def main():
    args = parse_args()
    device = get_device()

    # ディレクトリとTensorBoardライターの準備
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    if os.path.exists(log_dir):
        print(f"古いログディレクトリ {log_dir} を削除しています...")
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # データセットとデータローダー
    dataset = DicomPairDataset(hr_root_dir=args.hr_root, lr_root_dir=args.lr_root, patch_size_hr=args.patch_size_hr, scale=args.scale)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # モデル定義
    in_channels = 1
    # 注: models.pyのHRFlowのsplit_size=4に合わせて、z_cは12ch, z_hは4ch
    z_c_channels = (in_channels * 4 * 4) - 4 
    
    hr_flow = HRFlow().to(device)
    deg_flow = DegFlow(in_channels=in_channels, cond_channels=z_c_channels, n_gaussian=args.n_gaussian).to(device)
    lr_encoder = LREncoder(in_channels=in_channels, out_channels=z_c_channels, hidden_channels=args.hidden_channels).to(device)
    content_decoder = ContentDecoder(in_channels=z_c_channels, out_channels=in_channels, hidden_channels=args.hidden_channels).to(device)

    # ### 変更点: content_decoderのパラメータをオプティマイザの対象に追加 ###
    flow_params = list(hr_flow.parameters()) + list(deg_flow.parameters()) + \
                  list(lr_encoder.parameters()) + list(content_decoder.parameters())

    # ディスクリミネーター
    disc_content = ContentDiscriminator(in_channels=z_c_channels).to(device)
    disc_hr = ImageDiscriminator(in_channels=in_channels).to(device)
    disc_lr = ImageDiscriminator(in_channels=in_channels).to(device)
    disc_params = list(disc_content.parameters()) + list(disc_hr.parameters()) + list(disc_lr.parameters())

    # オプティマイザとスケジューラー
    optim_flow = torch.optim.Adam(flow_params, lr=args.lr, betas=(0.9, 0.999))
    optim_disc = torch.optim.Adam(disc_params, lr=args.lr_disc, betas=(0.9, 0.999))
    total_steps = len(dataloader) * args.epochs
    milestones = [int(total_steps * 0.5), int(total_steps * 0.75), int(total_steps * 0.9), int(total_steps * 0.95)]
    scheduler_flow = torch.optim.lr_scheduler.MultiStepLR(optim_flow, milestones=milestones, gamma=0.5)
    scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(optim_disc, milestones=milestones, gamma=0.5)

    # 損失関数
    likelihood_loss = LikelihoodLoss().to(device)
    latent_adv_loss = LatentAdversarialLoss().to(device)
    pixel_loss = PixelLoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    image_adv_loss = ImageAdversarialLoss().to(device)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for idx, (hr_patch, lr_patch) in enumerate(progress_bar):
            hr_patch = hr_patch.to(device)
            lr_patch = lr_patch.to(device)

            # =========================
            #    ジェネレータの学習
            # =========================
            optim_flow.zero_grad()
            
            # --- 1. 潜在変数の計算 ---
            z_c_hr, z_h, logdet_y = hr_flow(hr_patch, ldj=torch.zeros(hr_patch.size(0), device=device), reverse=False)
            z_c_lr = lr_encoder(lr_patch)
            z_x, logdet_x_given_y = deg_flow(lr_patch, ldj=torch.zeros(lr_patch.size(0), device=device), u=z_c_hr.detach(), reverse=False)
            
            # --- 2. 損失の計算 ---
            # NLL損失 (式16, 14)
            loss_nll = likelihood_loss(z_h, logdet_y) + likelihood_loss(z_x, logdet_x_given_y)
            
            # 潜在GAN損失 (式18のジェネレータ部分)
            loss_latent_gen = latent_adv_loss.generator_loss(disc_content, z_c_hr, z_c_lr)
            
            # ### 変更点: 式(17)のL_contentを完全に実装 ###
            # デコーダ f^-1_LR を使って、2つのz_cから画像を再構成
            reconst_from_hr = content_decoder(z_c_hr) # f^-1(z_HR_c, 0)
            reconst_from_lr = content_decoder(z_c_lr) # f^-1(z_LR_c, 0)
            
            # 比較対象となるターゲット画像を準備
            target_hr_downsampled = F.interpolate(hr_patch, scale_factor=1/args.scale, mode='bicubic', align_corners=False) # BDs(y)
            target_lr = lr_patch # x

            # 全ての画像をローパスフィルタ(LPF)にかける
            lpf_reconst_from_hr = gaussian_blur(reconst_from_hr, kernel_size=5)
            lpf_reconst_from_lr = gaussian_blur(reconst_from_lr, kernel_size=5)
            lpf_target_for_hr = gaussian_blur(target_hr_downsampled, kernel_size=5)
            lpf_target_for_lr = gaussian_blur(target_lr, kernel_size=5)

            # L_contentの4つの項を計算
            alpha = 0.05 # 論文より [cite: 437]
            # 項1: ||LPF(f^-1(z_HR_c)) - LPF(BDs(y))||_1
            loss_content_l1_hr = pixel_loss(lpf_reconst_from_hr, lpf_target_for_hr)
            # 項2: ||LPF(f^-1(z_LR_c)) - LPF(x)||_1
            loss_content_l1_lr = pixel_loss(lpf_reconst_from_lr, lpf_target_for_lr)
            # 項3: α * ||φ(f^-1(z_HR_c)) - φ(BDs(y))||_1
            loss_content_per_hr = perceptual_loss(reconst_from_hr, target_hr_downsampled)
            # 項4: α * ||φ(f^-1(z_LR_c)) - φ(x)||_1
            loss_content_per_lr = perceptual_loss(reconst_from_lr, target_lr)
            
            loss_content = (loss_content_l1_hr + loss_content_l1_lr) + alpha * (loss_content_per_hr + loss_content_per_lr)

            # --- 3. 論文の式(19)に従い、全体損失(Forward loss)を計算 ---
            # 論文の重み係数 (λ) は公開されていないため、一般的な値で設定
            loss_G = loss_nll + loss_content + loss_latent_gen

            # --- 4. 本学習フェーズ (Backward lossの追加) ---
            sr_image, ds_image = None, None
            if global_step >= args.pretrain_steps:
                sampled_h = torch.randn_like(z_h) * 0.8
                sr_input_z = [z_c_lr.detach(), sampled_h]
                sr_image, _ = hr_flow(sr_input_z, ldj=None, reverse=True)
                ds_image = content_decoder(z_c_hr.detach())

                # [cite_start]論文指定の重み λ1～λ6 [cite: 438]
                lambda_pix_ds = 0.5; lambda_per_ds = 0.5; lambda_gan_ds = 0.1
                lambda_pix_sr = 0.5; lambda_per_sr = 0.5; lambda_gan_sr = 0.1

                loss_adv_hr = image_adv_loss.generator_loss(disc_hr, fake=sr_image)
                loss_adv_lr = image_adv_loss.generator_loss(disc_lr, fake=ds_image)
                loss_pix_sr = pixel_loss(sr_image, hr_patch)
                loss_per_sr = perceptual_loss(sr_image, hr_patch)
                loss_pix_ds = pixel_loss(ds_image, target_hr_downsampled)
                loss_per_ds = perceptual_loss(ds_image, target_hr_downsampled)

                loss_G += (lambda_pix_ds * loss_pix_ds + lambda_per_ds * loss_per_ds + lambda_gan_ds * loss_adv_lr) + \
                          (lambda_pix_sr * loss_pix_sr + lambda_per_sr * loss_per_sr + lambda_gan_sr * loss_adv_hr)

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(flow_params, 1.0)
            optim_flow.step()

            # --- ディスクリミネータの学習 ---
            optim_disc.zero_grad()
            
            # 式(18) L_domain
            # 論文のアーキテクチャ(Fig. 3)ではz_dを別途計算してz_LR=z_c+z_dとするが、ここではz_c_lrで代用
            z_LR_substitute = z_c_lr.detach() 
            loss_dc = latent_adv_loss.disc_loss(disc_content, z_c_hr.detach(), z_c_lr.detach(), z_LR_substitute)
            loss_D = loss_dc

            if global_step >= args.pretrain_steps and sr_image is not None and ds_image is not None:
                loss_dhr = image_adv_loss.disc_loss(disc_hr, real=hr_patch, fake=sr_image.detach())
                loss_dlr = image_adv_loss.disc_loss(disc_lr, real=lr_patch, fake=ds_image.detach())
                loss_D += 0.5 * (loss_dhr + loss_dlr) # この0.5は一般的な重み
            
            loss_D.backward()
            optim_disc.step() 

            # --- ログ記録、チェックポイント保存 (変更なし) ---
            if global_step % args.log_interval == 0:
                writer.add_scalar("Loss/Gen_Total", loss_G.item(), global_step)
                writer.add_scalar("Loss/Disc_Total", loss_D.item(), global_step)
                writer.add_scalar("Loss_Comp/NLL", loss_nll.item(), global_step)
                writer.add_scalar("Loss_Comp/Content", loss_content.item(), global_step)
                writer.add_scalar("Loss_Comp/Latent_Adv", loss_latent_gen.item(), global_step)
                
                with torch.no_grad():
                    z_c_lr_vis = lr_encoder(lr_patch)
                    vis_sampled_h = torch.randn_like(z_h)
                    vis_sr_input = [z_c_lr_vis, vis_sampled_h]
                    sr_vis, _ = hr_flow(vis_sr_input, ldj=None, reverse=True)
                    writer.add_image("Images/A_Input_LR", lr_patch[0], global_step)
                    writer.add_image("Images/B_Output_SR", sr_vis[0], global_step)
                    writer.add_image("Images/C_GroundTruth_HR", hr_patch[0], global_step)
            
            global_step += 1
        
        scheduler_flow.step()
        scheduler_disc.step()
        
        with torch.no_grad():
            z_c_lr_vis = lr_encoder(lr_patch)
            vis_sampled_h = torch.randn_like(z_h)
            vis_sr_input = [z_c_lr_vis, vis_sampled_h]
            sr_vis, _ = hr_flow(vis_sr_input, ldj=None, reverse=True)
            save_visualizations(lr_patch[0], sr_vis[0].detach(), hr_patch[0], args.output_dir, epoch)
            
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