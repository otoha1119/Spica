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
    parser.add_argument("--output_dir", type=str, default="/workspace/checkpoints", help="Directory for checkpoints and logs")
    
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU (Paper: 32)")
    parser.add_argument("--patch_size_hr", type=int, default=192, help="Crop size for HR patches (Paper: 192)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for generator")
    parser.add_argument("--lr_disc", type=float, default=1e-5, help="Learning rate for discriminators")
    parser.add_argument("--scale", type=int, default=2, help="Super-resolution scale factor")
    parser.add_argument("--log_interval", type=int, default=100, help="Iterations between logging")
    parser.add_argument("--pretrain_steps", type=int, default=50000, help="Number of pre-training iterations (Paper: 50k)")
    
   
    parser.add_argument("--n_flows", type=int, default=4, help="Number of flow steps K in each FlowBlock (Paper: 16)")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels in Flow models")
    parser.add_argument("--n_gaussian", type=int, default=10, help="Number of gaussian mixtures for DegFlow")
    
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

    

    # Models
    in_channels = 1
    hr_flow = HRFlow(in_channels=in_channels, n_flows=args.n_flows,
                     hidden_channels=args.hidden_channels, scale=args.scale).to(device)

    content_decoder = ContentFlow().to(device)

    # z_h (3ch) を処理する専用モデル
    # 条件変数uであるz_c_hr(1ch)に合わせてcond_channels=1に修正
    deg_flow_h = DegFlow(in_channels=3, cond_channels=1,
                         n_gaussian=args.n_gaussian).to(device)
    
    # lr_encoderの出力チャンネル数を2に変更。z_c_lrとz_dが1chずつになるようにする
    lr_encoder = LRImageEncoderFM(in_channels=in_channels, out_channels=2).to(device)
    
    # z_d (1ch) を処理する専用モデル
    # 条件変数uであるz_c_lr(1ch)に合わせてcond_channels=1に修正
    deg_flow_d = DegFlow(in_channels=1, cond_channels=1,
                         n_gaussian=args.n_gaussian).to(device)

    # ジェネレータ関連の全モデルのパラメータを一つのリストにまとめる
    flow_params = list(hr_flow.parameters()) + list(content_decoder.parameters()) + \
                  list(deg_flow_h.parameters()) + list(deg_flow_d.parameters()) + \
                  list(lr_encoder.parameters())

    # ディスクリミネーター
    # z_c_hrとz_c_lrが共に1chになったため、in_channels=1で正しく動作する
    disc_content = ContentDiscriminator(in_channels=1).to(device)
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

            # =========================
            #    ジェネレータの学習
            # =========================
            optim_flow.zero_grad()
            
            # --- 1. 順伝播 ---
            """
            HR画像とLR画像を入力として、論文で定義されている
            様々な潜在変数 (z_c_hr, z_h, z_c_lr, z_d) と
            対数行列式 (logdet_hr, logdet_hf, logdet_deg) を計算
            """
            z_c_hr, z_h, logdet_hr = hr_flow(hr_patch, ldj=torch.zeros(hr_patch.size(0), device=device), reverse=False)
            
            z_lr_combined = lr_encoder(lr_patch)
            z_c_lr, z_d = torch.chunk(z_lr_combined, 2, dim=1)
            
            z_h_prime, logdet_hf = deg_flow_h(z_h, ldj=None, u=z_c_hr.detach(), reverse=False)
            z_d_prime, logdet_deg = deg_flow_d(z_d, ldj=None, u=z_c_lr.detach(), reverse=False)

            # --- 2. ジェネレータ損失の計算 ---
            # 式(14) NLL_y と 式(16) NLL_x 
            loss_nll = likelihood_loss(z_h_prime, logdet_hr + logdet_hf) + \
                       likelihood_loss(z_d_prime, logdet_deg)
            
            # 式(18) 潜在空間での敵対的損失を正しく計算しています。
            loss_latent_gen = latent_adv_loss.generator_loss(disc_content, z_c_hr, z_c_lr)
            
            # 式(17) L_content に対応? 
            reconst_lr_from_hr, _ = content_decoder(z_c_hr, ldj=None, reverse=True)
            reconst_lr_from_lr, _ = content_decoder(z_c_lr, ldj=None, reverse=True)
            target_lr_from_hr = F.interpolate(hr_patch, scale_factor=1/args.scale, mode='bicubic', align_corners=False)
            
            # LPFとしてガウシアンブラーを適用
            reconst_lr_from_hr_blur = gaussian_blur(reconst_lr_from_hr, kernel_size=5)
            reconst_lr_from_lr_blur = gaussian_blur(reconst_lr_from_lr, kernel_size=5)
            target_lr_from_hr_blur = gaussian_blur(target_lr_from_hr, kernel_size=5)
            lr_patch_blur = gaussian_blur(lr_patch, kernel_size=5)
            
            # ピクセル損失部分の計算
            loss_content_pix = pixel_loss(reconst_lr_from_hr_blur, target_lr_from_hr_blur) + \
                               pixel_loss(reconst_lr_from_lr_blur, lr_patch_blur)
            
            #知覚損失部分の計算
            loss_content_per = perceptual_loss(reconst_lr_from_hr, target_lr_from_hr) + \
                               perceptual_loss(reconst_lr_from_lr, lr_patch)
            
            #式(17) 重みα=0.05を掛けて合計
            loss_content = loss_content_pix + 0.05 * loss_content_per
            
            # --- 3. 学習戦略の適用 (事前学習 vs 本学習) ---
            # 式(19) 全体に対応 
            
            # 各項の重みを設定
            lambda_nll = 1.0         # 基準となる尤度損失
            lambda_content = 10.0    # 見た目の構造を重視するため、強めに設定
            lambda_domain = 0.05     # 学習を安定させるため、控えめに設定
            loss_G = (lambda_nll * loss_nll) + \
                    (lambda_content * loss_content) + \
                    (lambda_domain * loss_latent_gen)
                    
            sr_rand, ds_rand = None, None

            if global_step >= args.pretrain_steps:
                # --- ステージ2：本学習 ---
                z_h_sampled, _ = deg_flow_h(None, u=z_c_lr.detach(), reverse=True, tau=0.8)
                sr_rand, _ = hr_flow(z_c_lr.detach(), ldj=None, u_hf=z_h_sampled, reverse=True)
                ds_rand, _ = content_decoder(z_c_hr.detach(), ldj=None, reverse=True)
                
                # L_pixの計算
                loss_pix_sr = pixel_loss(sr_rand, F.interpolate(lr_patch, scale_factor=args.scale, mode='bicubic', align_corners=False))
                loss_pix_ds = pixel_loss(ds_rand, lr_patch)
                
                # L_perの計算
                loss_per_sr = perceptual_loss(sr_rand, hr_patch) # SR画像と本物のHR画像を比較
                loss_per_ds = perceptual_loss(ds_rand, lr_patch) # DS画像と本物のLR画像を比較
                
                # L_GANの計算
                loss_adv_hr = image_adv_loss.generator_loss(disc_hr, fake=sr_rand)
                loss_adv_lr = image_adv_loss.generator_loss(disc_lr, fake=ds_rand)
                
                # 論文指定の重み λ_per=0.5 を使って、知覚損失を加算する
                loss_G += 0.5 * (loss_pix_sr + loss_pix_ds) + \
                          0.5 * (loss_per_sr + loss_per_ds) + \
                          0.1 * (loss_adv_hr + loss_adv_lr)
                          

            # --- 4. ジェネレータのパラメータ更新 ---
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(flow_params, 1.0)
            optim_flow.step()

            # =========================
            #  ディスクリミネーターの学習
            # =========================
            optim_disc.zero_grad()
            
            # 1. まず、どのステージでも共通の latent loss (loss_dc) を計算し、
            #    loss_D を無条件に初期化する
            with torch.no_grad():
                z_LR = z_c_lr + z_d
            loss_dc = latent_adv_loss.disc_loss(disc_content, z_c_hr.detach(), z_c_lr.detach(), z_LR.detach())
            loss_D = loss_dc

            # 2. 本学習ステージに入ったら、追加で image loss を計算して loss_D に加算する
            if global_step >= args.pretrain_steps and sr_rand is not None and ds_rand is not None:
                loss_dhr = image_adv_loss.disc_loss(disc_hr, real=hr_patch, fake=sr_rand.detach())
                loss_dlr = image_adv_loss.disc_loss(disc_lr, real=lr_patch, fake=ds_rand.detach())
                loss_D += 0.5 * (loss_dhr + loss_dlr)
            
            # 3. 最後に、初期化された loss_D で backprop を行う
            loss_D.backward()
            optim_disc.step()
            
            # =========================
            #      ログ記録
            # =========================
            if global_step % args.log_interval == 0:
                with torch.no_grad():
                    # temp= を tau= に修正
                    z_c_lr_vis, _ = torch.chunk(lr_encoder(lr_patch), 2, dim=1)
                    z_h_sampled, _ = deg_flow_h(None, u=z_c_lr_vis, reverse=True, tau=0.0)
                    sr_vis, _ = hr_flow(z_c_lr_vis, ldj=None, u_hf=z_h_sampled, reverse=True)

                    writer.add_image("A_Input/LR", lr_patch[0], global_step)
                    writer.add_image("B_Output/SR_vis", sr_vis[0].detach(), global_step)
                    writer.add_image("C_GroundTruth/HR", hr_patch[0], global_step)
                    writer.add_histogram("Value_Distribution/LR_Input", lr_patch, global_step)
                    writer.add_histogram("Value_Distribution/SR_vis", sr_vis, global_step)
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

            # 学習ループ内と同じ、正しい手順で可視化用画像を生成する
            # 1. まず結合された潜在変数を受け取る
            z_lr_combined_vis = lr_encoder(vis_lr)
            # 2. 次に2つに分割する
            z_c_lr_vis, _ = torch.chunk(z_lr_combined_vis, 2, dim=1)
            
            # 3. 正しいモデルと引数でSR画像を生成する
            z_h_sampled, _ = deg_flow_h(None, u=z_c_lr_vis, reverse=True, tau=0.0)
            vis_sr, _ = hr_flow(z_c_lr_vis, ldj=None, u_hf=z_h_sampled, reverse=True)

            save_visualizations(vis_lr[0], vis_sr[0].detach(), vis_hr[0],
                                args.output_dir, epoch)
            
        state_to_save = {
            'hr_flow': hr_flow.state_dict(),
            'content_decoder': content_decoder.state_dict(),
            'deg_flow_h': deg_flow_h.state_dict(),
            'deg_flow_d': deg_flow_d.state_dict(),
            'lr_encoder': lr_encoder.state_dict(),
        }

        # PyTorchの標準機能で、辞書をファイルに保存
        torch.save(state_to_save, checkpoint_path)
        print(f"\n[Epoch {epoch}] Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()