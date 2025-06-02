###############################################
# sdflow/loss_sdflow.py
###############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Gaussian log probability
def gaussian_logprob(z, mean, logvar):
    return -0.5 * (logvar + ((z - mean) ** 2) / torch.exp(logvar) + torch.log(2 * torch.pi))

class LikelihoodLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, logdet):
        mean = torch.zeros_like(z)
        logvar = torch.zeros_like(z)
        log_p = gaussian_logprob(z, mean, logvar).sum(dim=[1,2,3])
        loss = - (log_p + logdet).mean()
        return loss

class LatentAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def generator_loss(self, D, z_c_hr, z_c_lr):
        pred_hr = D(z_c_hr)
        pred_lr = D(z_c_lr)
        loss_hr = self.mse(pred_hr, torch.zeros_like(pred_hr))
        loss_lr = self.mse(pred_lr, torch.ones_like(pred_lr))
        return 0.5 * (loss_hr + loss_lr)

    def disc_loss(self, D, z_c_hr, z_c_lr):
        real_pred = D(z_c_hr)
        fake_pred = D(z_c_lr)
        loss_real = self.mse(real_pred, torch.ones_like(real_pred))
        loss_fake = self.mse(fake_pred, torch.zeros_like(fake_pred))
        return 0.5 * (loss_real + loss_fake)

class PixelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        return self.criterion(pred, target)

class PerceptualLoss(nn.Module):
    def __init__(self, layer_idx=21):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:layer_idx]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        feat_pred = self.feature_extractor(pred.repeat(1,3,1,1))
        feat_target = self.feature_extractor(target.repeat(1,3,1,1))
        return F.mse_loss(feat_pred, feat_target)

class ImageAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def generator_loss(self, D, fake):
        pred_fake = D(fake)
        return self.mse(pred_fake, torch.ones_like(pred_fake))

    def disc_loss(self, D, real, fake):
        pred_real = D(real)
        pred_fake = D(fake)
        loss_real = self.mse(pred_real, torch.ones_like(pred_real))
        loss_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
        return 0.5 * (loss_real + loss_fake)
