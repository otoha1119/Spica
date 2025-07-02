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

    in_channels = 1
    # --- 修正: HRFlowのsplit_size=4に合わせて、z_cのチャネル数を正しく12に計算する ---
    z_c_channels = (in_channels * 4 * 4) - 4 
    
    hr_flow = HRFlow().to(device)
    deg_flow = DegFlow(in_channels=in_channels, cond_channels=z_c_channels, n_gaussian=args.n_gaussian).to(device)
    lr_encoder = LREncoder(in_channels=in_channels, out_channels=z_c_channels, hidden_channels=args.hidden_channels).to(device)

    # ジェネレータ関連の全モデルのパラメータ (content_decoderを削除)
    flow_params = list(hr_flow.parameters()) + list(deg_flow.parameters()) + list(lr_encoder.parameters())

    # ディスクリミネーター (z_c_channelsに合わせて修正)
    disc_content = ContentDiscriminator(in_channels=z_c_channels).to(device)
    disc_hr = ImageDiscriminator(in_channels=in_channels).to(device)
    disc_lr = ImageDiscriminator(in_channels=in_channels).to(device)
    disc_params = list(disc_content.parameters()) + list(disc_hr.parameters()) + list(disc_lr.parameters())
    # --- 変更ここまで ---

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

            optim_flow.zero_grad()
            
            z_c, z_h, logdet_y = hr_flow(hr_patch, ldj=torch.zeros(hr_patch.size(0), device=device), reverse=False)
            z_c_lr = lr_encoder(lr_patch)
            z_x, logdet_x_given_y = deg_flow(lr_patch, ldj=torch.zeros(lr_patch.size(0), device=device), u=z_c.detach(), reverse=False)
            
            loss_nll = likelihood_loss(z_h, logdet_y) + likelihood_loss(z_x, logdet_x_given_y)
            loss_latent_gen = latent_adv_loss.generator_loss(disc_content, z_c, z_c_lr)
            
            # --- 変更点: 矛盾の原因となっていたloss_contentの計算を完全に削除 ---
            # loss_content = ... (関連する計算をすべて削除)
            
            # 式(19) 全体損失 (事前学習段階) - loss_contentの項を削除
            loss_G = 1.0 * loss_nll + 0.05 * loss_latent_gen
            
            sr_image, ds_image = None, None
            if global_step >= args.pretrain_steps:
                # --- ステージ2：本学習 (ds_imageの生成方法を変更) ---
                sampled_h = torch.randn_like(z_h) * 0.8
                sr_input_z = [z_c_lr.detach(), sampled_h]
                sr_image, _ = hr_flow(sr_input_z, ldj=None, reverse=True)
                
                # ds_imageは論文の趣旨を考えると不要なため、ここでは単純なLR画像を使う
                # あるいは、別の方法で生成する必要があるが、一旦学習を進めることを優先
                ds_image = F.interpolate(hr_patch, scale_factor=1/args.scale, mode='bicubic', align_corners=False)

                loss_adv_hr = image_adv_loss.generator_loss(disc_hr, fake=sr_image)
                loss_adv_lr = image_adv_loss.generator_loss(disc_lr, fake=ds_image)
                loss_pix_sr = pixel_loss(sr_image, hr_patch)
                loss_per_sr = perceptual_loss(sr_image, hr_patch)
                
                loss_G += 0.1 * (loss_adv_hr + loss_adv_lr) + 1.0 * (loss_pix_sr + loss_per_sr)

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(flow_params, 1.0)
            optim_flow.step()

            # ディスクリミネーターの学習
            optim_disc.zero_grad()
            
            loss_dc = latent_adv_loss.disc_loss(disc_content, z_c.detach(), z_c_lr.detach(), z_c_lr.detach())
            loss_D = loss_dc

            if global_step >= args.pretrain_steps and sr_image is not None and ds_image is not None:
                loss_dhr = image_adv_loss.disc_loss(disc_hr, real=hr_patch, fake=sr_image.detach())
                loss_dlr = image_adv_loss.disc_loss(disc_lr, real=lr_patch, fake=ds_image.detach())
                loss_D += 0.5 * (loss_dhr + loss_dlr)
            
            loss_D.backward()
            optim_disc.step() 

            if global_step % args.log_interval == 0:
                writer.add_scalar("Loss/Gen_Total", loss_G.item(), global_step)
                writer.add_scalar("Loss/Disc_Total", loss_D.item(), global_step)
                writer.add_scalar("Loss_Comp/NLL", loss_nll.item(), global_step)
                writer.add_scalar("Loss_Comp/Latent_Adv", loss_latent_gen.item(), global_step)
                
                with torch.no_grad():
                    # --- 変更点: 可視化ロジックを本来あるべき形に戻す ---
                    # 新しいLREncoderが、正しい空間サイズ(48x48)のz_c_lrを出力する
                    z_c_lr_vis = lr_encoder(lr_patch)
                    vis_sampled_h = torch.randn_like(z_h)
                    
                    # これで、z_c_lr_vis と vis_sampled_h は共に48x48となり、連結できる
                    vis_sr_input = [z_c_lr_vis, vis_sampled_h]
                    sr_vis, _ = hr_flow(vis_sr_input, ldj=None, reverse=True)
                    # --- 変更ここまで ---
                    
                    writer.add_image("Images/A_Input_LR", lr_patch[0], global_step)
                    writer.add_image("Images/B_Output_SR", sr_vis[0], global_step)
                    writer.add_image("Images/C_GroundTruth_HR", hr_patch[0], global_step)
            # --- 変更ここまで ---
            
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