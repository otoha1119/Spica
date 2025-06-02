###############################################################
# loss.py
###############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def gaussian_logprob(z, mean, logvar):
    # z, mean, logvar: (B, C)
    return -0.5 * (logvar + ((z - mean) ** 2) / torch.exp(logvar) + torch.log(2 * torch.pi))


class LikelihoodLoss(nn.Module):
    def __init__(self):
        super(LikelihoodLoss, self).__init__()
        # 特に初期化不要

    def hr_loss(self, z_h, logdet):
        # z_h: 高周波潜在, logdet: Jacobianの行列式ログ和
        # ここでは標準正規分布を仮定
        mean = torch.zeros_like(z_h)
        logvar = torch.zeros_like(z_h)
        log_p = gaussian_logprob(z_h, mean, logvar).sum(dim=[1,2,3])
        loss = - (log_p + logdet).mean()
        return loss

    def lr_loss(self, z_d, logdet):
        # z_d: 劣化潜在, logdet: Jacobian
        # 混合ガウスの尤度は別途計算、ここでは簡易に標準正規で近似
        mean = torch.zeros_like(z_d)
        logvar = torch.zeros_like(z_d)
        log_p = gaussian_logprob(z_d, mean, logvar).sum(dim=[1,2,3])
        loss = - (log_p + logdet).mean()
        return loss


class PixelLoss(nn.Module):
    def __init__(self):
        super(PixelLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        return self.criterion(pred, target)


class PerceptualLoss(nn.Module):
    def __init__(self, layer_idx=21):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:layer_idx]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        feat_pred = self.feature_extractor(pred.repeat(1,3,1,1))
        feat_target = self.feature_extractor(target.repeat(1,3,1,1))
        return F.mse_loss(feat_pred, feat_target)


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.mse = nn.MSELoss()

    def discriminator_loss_image(self, D, real, fake):
        # LSGAN: real->1, fake->0
        real_pred = D(real)
        fake_pred = D(fake)
        loss_real = self.mse(real_pred, torch.ones_like(real_pred))
        loss_fake = self.mse(fake_pred, torch.zeros_like(fake_pred))
        return (loss_real + loss_fake) * 0.5

    def generator_loss_image(self, D, fake):
        fake_pred = D(fake)
        return self.mse(fake_pred, torch.ones_like(fake_pred))

    def discriminator_loss_latent(self, D, z_c_hr, z_c_lr):
        real_pred = D(z_c_hr)
        fake_pred = D(z_c_lr)
        loss_real = self.mse(real_pred, torch.ones_like(real_pred))
        loss_fake = self.mse(fake_pred, torch.zeros_like(fake_pred))
        return (loss_real + loss_fake) * 0.5

    def generator_loss_latent(self, D, z_c_hr, z_c_lr):
        # 逆にLR->HR, HR->LRが同じ分布になるよう
        pred_hr = D(z_c_hr)
        pred_lr = D(z_c_lr)
        loss_hr = self.mse(pred_hr, torch.zeros_like(pred_hr))
        loss_lr = self.mse(pred_lr, torch.ones_like(pred_lr))
        return (loss_hr + loss_lr) * 0.5


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        # VGG特徴抽出器
        vgg = models.vgg19(pretrained=True).features[:23].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.MSELoss()

    def forward(self, hr, ds_mean, z_c_hr, model):
        # hr: HR画像, ds_mean: HR->LR生成（平均像）
        # z_c_hrとmodelを利用して、HR由来z_cから生成したLRにも適用
        # ここでは簡易: ds_mean vs バイキュービックダウンsample(hr)
        hr_down = F.interpolate(hr, scale_factor=0.5, mode='bilinear', align_corners=False)
        loss_pix = F.mse_loss(ds_mean, hr_down)
        feat_ds = self.vgg(ds_mean.repeat(1,3,1,1))
        feat_hr_down = self.vgg(hr_down.repeat(1,3,1,1))
        loss_per = self.criterion(feat_ds, feat_hr_down)
        return loss_pix + loss_per
