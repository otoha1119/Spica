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
from sdflow.Dataset import DicomPairDataset
from sdflow.models import HRFlow, ContentFlow, DegFlow
from sdflow.discriminators import ContentDiscriminator, ImageDiscriminator
from sdflow.loss_sdflow import LikelihoodLoss, LatentAdversarialLoss, PixelLoss, PerceptualLoss, ImageAdversarialLoss
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
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        )
    def forward(self, x):
        return self.encoder(x)
# --- 変更ここまで ---


def main():
    args = parse_args()
    device = get_device()

    # ディレクトリとTensorBoardライターの準備 (変更なし)
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    if os.path.exists(log_dir):
        print(f"古いログディレクトリ {log_dir} を削除しています...")
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # データセットとデータローダー (変更なし)
    dataset = DicomPairDataset(hr_root_dir=args.hr_root, lr_root_dir=args.lr_root, patch_size_hr=args.patch_size_hr, scale=args.scale)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # --- 変更点: モデル定義を論文の構成に合わせてシンプル化 ---
    in_channels = 1
    # HRFlowの出力チャネル数を計算 (squeeze(1->4)*squeeze(4->16) - split(3) = 13)
    # models.pyのsplit_size=3を維持しているためz_cは13ch, z_hは3ch
    z_c_channels = (in_channels * 4 * 4) - 3 
    z_h_channels = 3
    
    hr_flow = HRFlow().to(device)
    # ContentFlowは論文のL_content(コンテンツ再構成損失)を計算するために使用
    content_decoder = ContentFlow().to(device)
    # DegFlow: lr_patch(1ch)を、hr_flowから得たz_c(13ch)を条件として尤度計算
    deg_flow = DegFlow(in_channels=in_channels, cond_channels=z_c_channels, 
                       n_gaussian=args.n_gaussian, hidden_channels=args.hidden_channels).to(device)
    # LREncoder: LR画像からz_c_hrと同じ次元の潜在変数z_c_lrを生成
    lr_encoder = LREncoder(in_channels=in_channels, out_channels=z_c_channels, 
                           hidden_channels=args.hidden_channels).to(device)

    # ジェネレータ関連の全モデルのパラメータ
    flow_params = list(hr_flow.parameters()) + list(content_decoder.parameters()) + \
                  list(deg_flow.parameters()) + list(lr_encoder.parameters())

    # ディスクリミネーター (論文通り3種類)
    disc_content = ContentDiscriminator(in_channels=z_c_channels).to(device)
    disc_hr = ImageDiscriminator(in_channels=in_channels).to(device)
    disc_lr = ImageDiscriminator(in_channels=in_channels).to(device)
    disc_params = list(disc_content.parameters()) + list(disc_hr.parameters()) + list(disc_lr.parameters())
    # --- 変更ここまで ---

    # オプティマイザとスケジューラー (変更なし)
    optim_flow = torch.optim.Adam(flow_params, lr=args.lr, betas=(0.9, 0.999))
    optim_disc = torch.optim.Adam(disc_params, lr=args.lr_disc, betas=(0.9, 0.999))
    total_steps = len(dataloader) * args.epochs
    milestones = [int(total_steps * 0.5), int(total_steps * 0.75), int(total_steps * 0.9), int(total_steps * 0.95)]
    scheduler_flow = torch.optim.lr_scheduler.MultiStepLR(optim_flow, milestones=milestones, gamma=0.5)
    scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(optim_disc, milestones=milestones, gamma=0.5)

    # 損失関数 (変更なし)
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
            
            # --- 変更点: 順伝播のロジックを論文に合わせて修正 ---
            # 1. HR画像からコンテンツ潜在変数z_cと確率潜在変数z_hを抽出 (log p(y))
            z_c, z_h, logdet_y = hr_flow(hr_patch, ldj=torch.zeros(hr_patch.size(0), device=device), reverse=False)
            
            # 2. LR画像からコンテンツ潜在変数z_c_lrを抽出 (Latent GAN用)
            z_c_lr = lr_encoder(lr_patch)

            # 3. 劣化モデルの尤度を計算 (log p(x|y))
            #    lr_patchを入力とし、HR由来のz_cを条件とする
            _, logdet_x_given_y = deg_flow(lr_patch, ldj=torch.zeros(lr_patch.size(0), device=device), u=z_c.detach(), reverse=False)
            
            # --- ジェネレータ損失の計算 ---
            # 式(14) 負の対数尤度損失
            loss_nll = likelihood_loss(logdet_y) + likelihood_loss(logdet_x_given_y)
            
            # 式(18) 潜在空間での敵対的損失
            loss_latent_gen = latent_adv_loss.generator_loss(disc_content, real_latent=z_c, fake_latent=z_c_lr)
            
            # 式(17) コンテンツ再構成損失
            reconst_lr, _ = content_decoder(z_c, ldj=None, reverse=True)
            # ガウシアンブラーを削除し、論文の意図通り直接比較
            loss_content_pix = pixel_loss(reconst_lr, F.interpolate(hr_patch, scale_factor=1/args.scale, mode='bicubic', align_corners=False))
            loss_content_per = perceptual_loss(reconst_lr, F.interpolate(hr_patch, scale_factor=1/args.scale, mode='bicubic', align_corners=False))
            loss_content = loss_content_pix + 0.05 * loss_content_per

            # 式(19) 全体損失 (事前学習段階)
            # 論文の重みに従う (λ_nll=1, λ_content=10, λ_domain=0.05)
            loss_G = 1.0 * loss_nll + 10.0 * loss_content + 0.05 * loss_latent_gen
            
            sr_image, ds_image = None, None # スコープを広げる
            if global_step >= args.pretrain_steps:
                # --- ステージ2：本学習 (GAN損失の追加) ---
                # z_hを正規分布からサンプリング
                sampled_h = torch.randn_like(z_h) * 0.8 # tau=0.8
                # z_c_lrとsampled_hからSR画像を生成
                sr_input_z = [z_c_lr.detach(), sampled_h]
                sr_image, _ = hr_flow(sr_input_z, ldj=None, reverse=True)
                
                # z_c_hrからDS画像を生成 (CycleGAN的な損失)
                ds_image, _ = content_decoder(z_c.detach(), ldj=None, reverse=True)
                
                # 画像GAN損失
                loss_adv_hr = image_adv_loss.generator_loss(disc_hr, fake=sr_image)
                loss_adv_lr = image_adv_loss.generator_loss(disc_lr, fake=ds_image)
                # ピクセル・知覚損失
                loss_pix_sr = pixel_loss(sr_image, hr_patch)
                loss_per_sr = perceptual_loss(sr_image, hr_patch)
                
                # 論文の重み λ_GAN=0.1, λ_rec=1.0 (L1とVGGをまとめたもの)
                loss_G += 0.1 * (loss_adv_hr + loss_adv_lr) + 1.0 * (loss_pix_sr + loss_per_sr)
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(flow_params, 1.0)
            optim_flow.step()

            # =========================
            #  ディスクリミネーターの学習
            # =========================
            optim_disc.zero_grad()
            
            # Latent GAN
            loss_dc = latent_adv_loss.disc_loss(disc_content, real_latent=z_c.detach(), fake_latent=z_c_lr.detach())
            loss_D = loss_dc

            if global_step >= args.pretrain_steps and sr_image is not None and ds_image is not None:
                # Image GAN
                loss_dhr = image_adv_loss.disc_loss(disc_hr, real=hr_patch, fake=sr_image.detach())
                loss_dlr = image_adv_loss.disc_loss(disc_lr, real=lr_patch, fake=ds_image.detach())
                loss_D += 0.5 * (loss_dhr + loss_dlr)
            
            loss_D.backward()
            optim_disc.step()
            # --- 変更ここまで ---
            
            # =========================
            #      ログ記録 (ほぼ変更なし)
            # =========================
            if global_step % args.log_interval == 0:
                writer.add_scalar("Loss/Gen_Total", loss_G.item(), global_step)
                writer.add_scalar("Loss/Disc_Total", loss_D.item(), global_step)
                writer.add_scalar("Loss_Comp/NLL", loss_nll.item(), global_step)
                writer.add_scalar("Loss_Comp/Content", loss_content.item(), global_step)
                writer.add_scalar("Loss_Comp/Latent_Adv", loss_latent_gen.item(), global_step)
                
                with torch.no_grad():
                    # 可視化用のSR画像生成
                    vis_z_c, vis_z_h, _ = hr_flow(hr_patch, ldj=None, reverse=False)
                    vis_sampled_h = torch.randn_like(vis_z_h) # tau=0でノイズなし
                    vis_sr_input = [vis_z_c, vis_sampled_h]
                    sr_vis, _ = hr_flow(vis_sr_input, ldj=None, reverse=True)
                    
                    writer.add_image("Images/A_Input_LR", lr_patch[0], global_step)
                    writer.add_image("Images/B_Output_SR", sr_vis[0], global_step)
                    writer.add_image("Images/C_GroundTruth_HR", hr_patch[0], global_step)

            global_step += 1
        
        scheduler_flow.step()
        scheduler_disc.step()
        
        # エポックごとのチェックポイント保存 (変更なし)
        # ... (チェックポイント保存ロジックは省略せずにここに記述) ...
        # (save_checkpoint関数を呼び出すように変更)
        checkpoint_path = os.path.join(args.output_dir, "checkpoints", f"epoch_{epoch}.pth")
        save_checkpoint(
            model={'hr_flow': hr_flow, 'content_decoder': content_decoder, 'deg_flow': deg_flow, 'lr_encoder': lr_encoder},
            optim_flow=optim_flow,
            optim_disc=optim_disc,
            epoch=epoch,
            path=checkpoint_path
        )
        print(f"\n[Epoch {epoch}] Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()