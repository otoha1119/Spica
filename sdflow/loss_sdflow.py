###############################################
# sdflow/loss_sdflow.py
###############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights
import math  # 追加で math モジュールを使用


# Gaussian log probability
def gaussian_logprob(z, mean, logvar):
    """
    z, mean, logvar はすべて同じデバイス・dtype の Tensor であることを前提とします。
    定数 2*pi も Tensor 化して torch.log に渡すよう修正。
    """
    # 定数 2*pi を Tensor として生成（z と同じデバイス・dtype）
    two_pi = torch.tensor(2 * math.pi, dtype=z.dtype, device=z.device)
    return -0.5 * (
        logvar
        + ((z - mean) ** 2) / torch.exp(logvar)
        + torch.log(two_pi)
    )


class LikelihoodLoss(nn.Module):
    """
    NLL (Negative Log Likelihood) 損失
    論文(14)式

    """
    def __init__(self):
        super().__init__()

    def forward(self, z, logdet):
        """
        Args:
            z (Tensor): Flow から得られた潜在変数テンソル (B, C, H, W)
            logdet (Tensor): Flow から得られたログデターミナント (B,)
        Returns:
            Tensor: バッチごとの平均 NLL スカラー
        """
        # 平均と対数分散をゼロテンソルとみなす
        mean = torch.zeros_like(z)
        logvar = torch.zeros_like(z)
        # ガウス対数確率を計算し、空間次元を sum
        log_p = gaussian_logprob(z, mean, logvar).sum(dim=[1, 2, 3])
        # NLL = - (log_p + logdet) をバッチ平均
        loss = - (log_p + logdet).mean()
        return loss



class LatentAdversarialLoss(nn.Module):
    """
    潜在空間 (z_c) に対する敵対的損失 (Adversarial Loss)。
    論文の式(18) L_domain に基づいて修正。
    """
    def __init__(self, beta1=0.05, beta2=0.5): 
        super().__init__()
        self.mse = nn.MSELoss()
        self.beta1 = beta1
        self.beta2 = beta2

    def generator_loss(self, D, z_c_hr, z_c_lr):
        """
        ジェネレータ側の損失。識別器を騙すことが目的。
        """
        pred_hr = D(z_c_hr)
        pred_lr = D(z_c_lr)
        loss_hr = self.mse(pred_hr, torch.ones_like(pred_hr))
        loss_lr = self.mse(pred_lr, torch.zeros_like(pred_lr))
        return 0.5 * (loss_hr + loss_lr) #0.5 で平均を取っている

    def disc_loss(self, D, z_c_hr, z_c_lr, z_LR): # z_LRを引数に追加
        """
        識別器側の損失。論文の式(18)を完全に実装。
        """
        real_pred = D(z_c_hr)
        fake_pred = D(z_c_lr)
        
        loss_real = self.mse(real_pred, torch.zeros_like(real_pred))
        loss_fake = self.mse(fake_pred, torch.ones_like(fake_pred))
        adv_loss = 0.5 * (loss_real + loss_fake)
        
        # beta1の正則化項
        l2_reg = z_c_hr.pow(2).mean() + z_c_lr.pow(2).mean()
        
        # beta2の正則化項 ||z_LR - sg(z_c^LR)||_2
        # sg(stop_gradient)は .detach() で実装
        reconstruction_reg = F.mse_loss(z_LR, z_c_lr.detach())

        # 3つの項を合計
        total_loss = adv_loss + self.beta1 * l2_reg + self.beta2 * reconstruction_reg
        
        return total_loss

class PixelLoss(nn.Module):
    """
    ピクセル空間での L1 損失 (L1 Loss)。
    画素単位での誤差を評価します。
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): 予測画像 (B, C, H, W)
            target (Tensor): 真の画像 (B, C, H, W)
        Returns:
            Tensor: バッチ平均の L1 損失
        """
        return self.criterion(pred, target)


class PerceptualLoss(nn.Module):
    """
    VGG19 を用いた知覚的損失 (Perceptual Loss)。
    VGG19 の特定層の特徴マップを抽出し、その差を L2 で計算します。
    """
    def __init__(self, layer_idx=21):
        super().__init__()
        # 事前学習済み VGG19 の特徴抽出部分を参照
        weights = VGG19_Weights.DEFAULT 
        vgg = models.vgg19(weights=weights).features
        # layer_idx 番目までのサブシーケンスを取得し eval モードに設定
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:layer_idx]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # 重みは固定

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): 予測画像 (B, 1, H, W)。VGG19 は 3 チャネル前提なので repeat で 3 チャネル化
            target (Tensor): 真の画像 (B, 1, H, W)
        Returns:
            Tensor: 知覚的損失スカラー
        """
        # 1チャネル→3チャネルに拡張
        feat_pred = self.feature_extractor(pred.repeat(1, 3, 1, 1))
        feat_target = self.feature_extractor(target.repeat(1, 3, 1, 1))
        return F.mse_loss(feat_pred, feat_target)


class ImageAdversarialLoss(nn.Module):
    """
    画像空間に対する敵対損失 (Adversarial Loss)。
    生成画像 fake を判別器 D に通し、ジェネレータと識別器の損失を計算します。
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def generator_loss(self, D, fake):
        """
        ジェネレータ損失: D(fake) → 1 (本物)
        Args:
            D (nn.Module): 画像判別器
            fake (Tensor): 生成画像 (B, C, H, W)
        Returns:
            Tensor: ジェネレータ損失スカラー
        """
        pred_fake = D(fake)
        return self.mse(pred_fake, torch.ones_like(pred_fake))

    def disc_loss(self, D, real, fake):
        """
        識別器損失: D(real) → 1 (本物), D(fake) → 0 (偽物)
        Args:
            D (nn.Module): 画像判別器
            real (Tensor): 真の画像 (B, C, H, W)
            fake (Tensor): 生成画像 (B, C, H, W)
        Returns:
            Tensor: 識別器損失スカラー
        """
        pred_real = D(real)
        pred_fake = D(fake)
        loss_real = self.mse(pred_real, torch.ones_like(pred_real))
        loss_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
        return 0.5 * (loss_real + loss_fake)
