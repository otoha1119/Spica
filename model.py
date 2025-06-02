###############################################################
# model.py
###############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# GitHubのSDFlow実装をインポートする前提
# パス例: sdflow以下に各モジュールが配置されている
from sdflow.modules.glow import FlowStep, Flow
from sdflow.modules.ops import ActNorm, InvConv
from sdflow.modules.coupling import AffineCoupling
from sdflow.modules.rrdb import RRDB


class HRFlow(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, n_levels=3, n_flow_steps=4):
        super(HRFlow, self).__init__()
        # GlowベースのFlowを構築
        self.flow = Flow(in_channels, hidden_channels, n_levels, n_flow_steps)

    def forward(self, x, reverse=False, z=None):
        # x: HR画像 (B, 1, H, W)
        # forward: 画像->潜在, reverse: 潜在->画像
        if not reverse:
            z, logdet = self.flow(x)
            # zをコンテンツz_cと高周波z_hに分割
            z_c, z_h = torch.chunk(z, 2, dim=1)
            return z_c, z_h, logdet
        else:
            # zは( z_c, z_h )を結合したもの
            return self.flow(z, reverse=True)


class LRFlow(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, n_levels=3, n_flow_steps=4):
        super(LRFlow, self).__init__()
        # LR画像用のFlowを構築 (DM ResBlockを組み込む)
        self.flow = Flow(in_channels, hidden_channels, n_levels, n_flow_steps)

    def forward(self, x, reverse=False, z=None):
        if not reverse:
            z, logdet = self.flow(x)
            z_c, z_d = torch.chunk(z, 2, dim=1)
            return z_c, z_d, logdet
        else:
            return self.flow(z, reverse=True)


class HFFlow(nn.Module):
    def __init__(self, z_c_channels=32, z_h_channels=32, n_blocks=8):
        super(HFFlow, self).__init__()
        # 条件付きフロー: Affine Coupling with RRDB条件ネット
        self.condition_net = self._make_rrdb(n_blocks, in_channels=z_c_channels)
        self.flow = Flow(z_h_channels * 2, z_h_channels, 1, 2)  # 簡易指定

    def _make_rrdb(self, n_blocks, in_channels):
        layers = []
        for _ in range(n_blocks):
            layers.append(RRDB(in_channels))
        return nn.Sequential(*layers)

    def forward(self, z_h, z_c, reverse=False, temp=1.0):
        # z_h: 高周波潜在, z_c: コンテンツ潜在
        cond = self.condition_net(z_c)
        if not reverse:
            # 条件付きエンコード: concat(z_h, cond)
            inp = torch.cat([z_h, cond], dim=1)
            z, logdet = self.flow(inp)
            # zをGaussianに合わせる
            return z, logdet
        else:
            # 逆変換: サンプリング用
            # temp で分布の広がりを調整
            mean = torch.zeros_like(z_h)
            eps = torch.randn_like(mean) * temp
            z_prior = mean + eps
            inp = torch.cat([z_prior, cond], dim=1)
            z_rev = self.flow(inp, reverse=True)
            return z_rev


class DegFlow(nn.Module):
    def __init__(self, z_c_channels=32, z_d_channels=32, n_blocks=4, n_components=16):
        super(DegFlow, self).__init__()
        self.condition_net = self._make_rrdb(n_blocks, in_channels=z_c_channels)
        # 混合ガウスの平均・分散を学習可能パラメータとして保持
        self.means = nn.Parameter(torch.zeros(n_components, z_d_channels))
        self.log_vars = nn.Parameter(torch.zeros(n_components, z_d_channels))
        self.flow = Flow(z_d_channels * 2, z_d_channels, 1, 2)

    def _make_rrdb(self, n_blocks, in_channels):
        layers = []
        for _ in range(n_blocks):
            layers.append(RRDB(in_channels))
        return nn.Sequential(*layers)

    def forward(self, z_d, z_c, reverse=False, temp=1.0):
        cond = self.condition_net(z_c)
        if not reverse:
            inp = torch.cat([z_d, cond], dim=1)
            z, logdet = self.flow(inp)
            # zを混合ガウスに合わせる: 実装は損失側で計算
            return z, logdet
        else:
            # サンプリング: 混合ガウスから選択してサンプル
            batch = z_c.size(0)
            # コンポーネントをランダムに選択
            comp_idx = torch.randint(0, self.means.size(0), (batch,), device=z_c.device)
            means = self.means[comp_idx]
            vars = torch.exp(self.log_vars[comp_idx])
            eps = torch.randn_like(means) * torch.sqrt(vars) * temp
            z_prior = means + eps
            inp = torch.cat([z_prior, cond], dim=1)
            z_rev = self.flow(inp, reverse=True)
            return z_rev


class SDFlowModel(nn.Module):
    def __init__(self):
        super(SDFlowModel, self).__init__()
        # チャネル数やレベル数は適宜調整
        self.hr_flow = HRFlow(in_channels=1, hidden_channels=64, n_levels=3, n_flow_steps=4)
        self.lr_flow = LRFlow(in_channels=1, hidden_channels=64, n_levels=3, n_flow_steps=4)
        self.hf_flow = HFFlow(z_c_channels=32, z_h_channels=32, n_blocks=8)
        self.deg_flow = DegFlow(z_c_channels=32, z_d_channels=32, n_blocks=4, n_components=16)

    def forward_hr(self, hr_img):
        # HR画像 -> z_c_hr, z_h  (正方向)
        z_c_hr, z_h, logdet_hr = self.hr_flow(hr_img)
        # z_h を条件付きフローにかけて z_h_trans
        z_h_trans, logdet_hf = self.hf_flow(z_h, z_c_hr)
        # 全ログデターミナントを合計
        logdet = logdet_hr + logdet_hf
        return z_c_hr, z_h_trans, None, None, logdet

    def forward_lr(self, lr_img):
        # LR画像 -> z_c_lr, z_d
        z_c_lr, z_d, logdet_lr = self.lr_flow(lr_img)
        z_d_trans, logdet_deg = self.deg_flow(z_d, z_c_lr)
        logdet = logdet_lr + logdet_deg
        return z_c_lr, z_d_trans, None, None, logdet

    def generate_sr(self, lr_img, temp=0.0):
        # LR -> z_c_lr, _
        z_c_lr, _, _ = self.lr_flow(lr_img)
        # z_h をtemp付きでサンプリング
        z_h = self.hf_flow(None, z_c_lr, reverse=True, temp=temp)
        # z_prior に z_c_lr と z_h を結合して HR Flow 逆方向
        z = torch.cat([z_c_lr, z_h], dim=1)
        sr = self.hr_flow(None, reverse=True, z=z)
        return sr

    def generate_ds(self, hr_img, temp=0.0):
        # HR -> z_c_hr, _
        z_c_hr, _, _ = self.hr_flow(hr_img)
        # z_d を temp 付きサンプリング
        z_d = self.deg_flow(None, z_c_hr, reverse=True, temp=temp)
        z = torch.cat([z_c_hr, z_d], dim=1)
        ds = self.lr_flow(None, reverse=True, z=z)
        return ds


class HRDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(HRDiscriminator, self).__init__()
        layers = []
        channels = in_channels
        for i in range(4):
            layers.append(nn.Conv2d(channels, 64 * (2 ** i), kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels = 64 * (2 ** i)
        layers.append(nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LRDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(LRDiscriminator, self).__init__()
        layers = []
        channels = in_channels
        for i in range(3):
            layers.append(nn.Conv2d(channels, 64 * (2 ** i), kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels = 64 * (2 ** i)
        layers.append(nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LatentDiscriminator(nn.Module):
    def __init__(self, in_channels=32):
        super(LatentDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, z_c):
        return self.model(z_c)