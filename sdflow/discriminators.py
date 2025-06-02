import torch
import torch.nn as nn

class ContentDiscriminator(nn.Module):
    """
    潜在空間 (content latent) に対して HR/LR を識別するディスクリミネータ
    入力: z_c (shape: [B, C, H, W])
    出力: スカラー (または小さなマップ) の判定値
    """
    def __init__(self, in_channels=32, base_channels=64):
        super(ContentDiscriminator, self).__init__()
        layers = []
        ch = in_channels
        for i in range(3):
            layers.append(nn.Conv2d(ch, base_channels * (2 ** i), kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = base_channels * (2 ** i)
        layers.append(nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=0))
        self.model = nn.Sequential(*layers)

    def forward(self, z_c):
        # z_c の空間次元を必要に応じて reshaped する場合はここで行う
        # 出力は [B,1,1,1] などを想定し、後で squeeze する
        out = self.model(z_c)
        return out.view(out.size(0), -1)

class ImageDiscriminator(nn.Module):
    """
    画像ドメイン (HR or LR) に対して判定する PatchGAN ライクなディスクリミネータ
    入力: 画像テンソル (shape: [B, 1, H, W])
    出力: マップ形式の判定値
    """
    def __init__(self, in_channels=1, base_channels=64):
        super(ImageDiscriminator, self).__init__()
        layers = []
        ch = in_channels
        for i in range(4):
            layers.append(nn.Conv2d(ch, base_channels * (2 ** i), kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = base_channels * (2 ** i)
        layers.append(nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        # 出力は [B,1,H_out,W_out] のスコアマップ
        return self.model(img)
