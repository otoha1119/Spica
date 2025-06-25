import os
import argparse
from tqdm import tqdm
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

# 公式コードのモジュールを正しくインポート
from sdflow.Dataset import DicomPairDataset
from sdflow.models import HRFlow, ContentFlow, DegFlow
from sdflow.modules import LRImageEncoderFM
from sdflow.discriminators import ContentDiscriminator, ImageDiscriminator
from sdflow.loss_sdflow import LikelihoodLoss, LatentAdversarialLoss, PixelLoss, PerceptualLoss, ImageAdversarialLoss
from sdflow.utils_sdflow import get_device, save_checkpoint, save_visualizations

def parse_args():
    parser = argparse.ArgumentParser(description="SDFlow Training - Paper Aligned")
    parser.add_argument("--lr_root", type=str, default="/workspace/DataSet/ImageCAS", help="LR DICOM directory")
    parser.add_argument("--hr_root", type=str, default="/workspace/DataSet/photonCT/PhotonCT1024v2", help="HR DICOM directory")
    parser.add_argument("--output_dir", type=str, default="/workspace/checkpoints_paper_ver", help="Directory for checkpoints and logs")
    
    # 論文とあなたの環境に合わせたパラメータ
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU (Paper: 32)")
    parser.add_argument("--patch_size_hr", type=int, default=192, help="Crop size for HR patches (Paper: 192)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for generator")
    parser.add_argument("--lr_disc", type=float, default=1e-5, help="Learning rate for discriminators")
    parser.add_argument("--scale", type=int, default=2, help="Super-resolution scale factor")
    parser.add_argument("--log_interval", type=int, default=100, help="Iterations between logging")
    parser.add_argument("--pretrain_steps", type=int, default=50000, help="Number of pre-training iterations (Paper: 50k)")
    
    return parser.parse_args()

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

    # --- モデルの初期化 (公式コードと論文の構成に準拠) ---
    in_channels = 1
    # 論文のアーキテクチャに合わせて、個別のモデルを定義
    # HRFlowはマルチスケール構造を持つ
    hr_flow = HRFlow().to(device)
    # LRFlowの役割は、この専用エンコーダが担う
    lr_encoder = LRImageEncoderFM(in_channels=in_channels, out_channels=in_channels * 2).to(device)
    # コンテンツz_cから画像を再構成するためのデコーダ
    content_decoder = ContentFlow().to(device)
    # 条件付きFlow（HF用とDegradation用）
    hf_flow = DegFlow(in_channels=in_channels, cond_channels=in_channels, n_gaussian=16).to(device)
    deg_flow = DegFlow(in_channels=in_channels, cond_channels=in_channels, n_gaussian=16).to(device)
    
    # ジェネレータ関連の全モデルのパラメータを一つのリストにまとめる
    flow_params = list(hr_flow.parameters()) + list(lr_encoder.parameters()) + list(content_decoder.parameters()) + \
                  list(hf_flow.parameters()) + list(deg_flow.parameters())

    # ディスクリミネーター
    disc_content = ContentDiscriminator(in_channels=in_channels).to(device)
    disc_hr = ImageDiscriminator(in_channels=in_channels).to(device)
    disc_lr = ImageDiscriminator(in_channels=in_channels).to(device)
    disc_params = list(disc_content.parameters()) + list(disc_hr.parameters()) + list(disc_lr.parameters())

    # オプティマイザ
    optim_flow = torch.optim.Adam(flow_params, lr=args.lr, betas=(0.9, 0.999))
    optim_disc = torch.optim.Adam(disc_params, lr=args.lr_disc, betas=(0.9, 0.999))

    # スケジューラー (論文に合わせて変更)
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

            # --- ジェネレータの学習 ---
            optim_flow.zero_grad()
            
            # 論文のアーキテクチャに沿ったフォワードパス
            z_c_hr, z_h, logdet_hr = hr_flow(hr_patch, ldj=None, reverse=False)
            z_c_lr, z_d = lr_encoder(lr_patch)
            _, logdet_hf = hf_flow(z_h, ldj=None, u=z_c_hr.detach(), reverse=False)
            _, logdet_deg = deg_flow(z_d, ldj=None, u=z_c_lr.detach(), reverse=False)

            # 1. NLL損失 (L_nll)
            loss_nll = (logdet_hr.mean() + logdet_hf.mean() + logdet_deg.mean()) * -1

            # 2. 潜在空間の敵対的損失 (L_domain)
            loss_latent_gen = latent_adv_loss.generator_loss(disc_content, z_c_hr, z_c_lr)

            # 3. コンテンツ損失 (L_content)
            reconst_lr_from_hr, _ = content_decoder(z_c_hr, ldj=None, reverse=True)
            reconst_lr_from_lr, _ = content_decoder(z_c_lr, ldj=None, reverse=True)
            
            target_lr_from_hr = F.interpolate(hr_patch, scale_factor=1/args.scale, mode='bicubic', align_corners=False)
            
            # ローパスフィルタとしてガウシアンブラーを適用
            reconst_lr_from_hr_blur = gaussian_blur(reconst_lr_from_hr, kernel_size=5)
            reconst_lr_from_lr_blur = gaussian_blur(reconst_lr_from_lr, kernel_size=5)
            target_lr_from_hr_blur = gaussian_blur(target_lr_from_hr, kernel_size=5)
            lr_patch_blur = gaussian_blur(lr_patch, kernel_size=5)

            loss_content_pix = pixel_loss(reconst_lr_from_hr_blur, target_lr_from_hr_blur) + \
                               pixel_loss(reconst_lr_from_lr_blur, lr_patch_blur)
            loss_content_per = perceptual_loss(reconst_lr_from_hr, target_lr_from_hr) + \
                               perceptual_loss(reconst_lr_from_lr, lr_patch)
            loss_content = loss_content_pix + 0.05 * loss_content_per
            
            # ジェネレータの基本損失 (事前学習で使う)
            loss_G = loss_nll + loss_content + 0.05 * loss_latent_gen

            # --- 学習戦略の適用 ---
            if global_step >= args.pretrain_steps: # 本学習フェーズ
                # 逆方向の生成と、それに対する損失を追加
                sr_rand, _ = hr_flow(z_c_lr.detach(), ldj=None, u_hf=hf_flow(None, u=z_c_lr.detach(), reverse=True), reverse=True)
                ds_rand = lr_encoder.reverse(z_c_hr.detach(), z_d_u=deg_flow(None, u=z_c_hr.detach(), reverse=True))

                loss_adv_hr = image_adv_loss.generator_loss(disc_hr, fake=sr_rand)
                loss_adv_lr = image_adv_loss.generator_loss(disc_lr, fake=ds_rand)
                
                loss_pix_sr = pixel_loss(sr_rand, F.interpolate(lr_patch, scale_factor=args.scale, align_corners=False))
                loss_pix_ds = pixel_loss(ds_rand, lr_patch)
                
                loss_G += 0.1 * (loss_adv_hr + loss_adv_lr) + 0.5 * (loss_pix_sr + loss_pix_ds)

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(flow_params, 1.0)
            optim_flow.step()

            # --- ディスクリミネーターの学習 ---
            optim_disc.zero_grad()
            loss_dc = latent_adv_loss.disc_loss(disc_content, z_c_hr.detach(), z_c_lr.detach())
            loss_D = loss_dc

            if global_step >= args.pretrain_steps:
                loss_dhr = image_adv_loss.disc_loss(disc_hr, real=hr_patch, fake=sr_rand.detach())
                loss_dlr = image_adv_loss.disc_loss(disc_lr, real=lr_patch, fake=ds_rand.detach())
                loss_D += 0.5 * (loss_dhr + loss_dlr)
            
            loss_D.backward()
            optim_disc.step()
            
            if global_step % args.log_interval == 0:
                with torch.no_grad():
                    # 1. 各画像を個別に、それぞれの解像度のまま記録する
                    # タグの先頭にアルファベットを付けて表示順をコントロール (A→B→C)
                    writer.add_image("A_Input/LR", lr_patch[0], global_step)
                    writer.add_image("B_Output/SR", sr_rand[0].detach(), global_step)
                    writer.add_image("C_GroundTruth/HR", hr_patch[0], global_step)

                    # 2. 画像の「細かい値の分布」をヒストグラムとして記録する
                    writer.add_histogram("Value_Distribution/LR_Input", lr_patch, global_step)
                    writer.add_histogram("Value_Distribution/SR_Output", sr_rand, global_step)

                    # 3. Lossグラフの表示
                    writer.add_scalar("D_Losses/Gen_Total", loss_G.item(), global_step)
                    writer.add_scalar("D_Losses/Disc_Total", loss_D.item(), global_step)
                    writer.add_scalar("D_Losses/NLL", loss_nll.item(), global_step)
                    writer.add_scalar("D_Losses/Content", loss_content.item(), global_step)

            global_step += 1
        
        scheduler_flow.step()
        scheduler_disc.step()
        
        with torch.no_grad():
            # 1番目のデータで可視化サンプルを生成
            # データセットから直接取得するのではなく、現在のバッチのデータを使う
            if 'vis_lr' not in locals() and 'vis_hr' not in locals(): # 最初のバッチを保存
                vis_lr = lr_patch
                vis_hr = hr_patch

            # --- 新しいモデル構成に合わせた呼び出し方に修正 ---
            z_c_lr, _ = lr_encoder(vis_lr)
            z_h_sampled = hf_flow(None, u=z_c_lr, reverse=True, temp=0.0) # temp=0.0で決定論的出力を得る
            vis_sr, _ = hr_flow(z_c_lr, ldj=None, u_hf=z_h_sampled, reverse=True)
            # --- 修正ここまで ---

            save_visualizations(vis_lr[0], vis_sr[0].detach(), vis_hr[0],
                                args.output_dir, epoch)

if __name__ == "__main__":
    main()