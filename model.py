# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ① Glow/RealNVPベースの FlowStep/Flow を sdflow からインポート
from sdflow.modules.glow import FlowStep, Flow

# ② ActNorm, InvConv, AffineCoupling を sdflow.modules からインポート
from sdflow.modules.ops import ActNorm, InvConv
from sdflow.modules.coupling import AffineCoupling

# ③ 条件付きフロー用のネットワーク（RRDB や module_util）を sdflow.conditional_net 経由でインポート
from sdflow.conditional_net import module_util as mutil
from sdflow.conditional_net.RRDBNet import RRDB

################################################################################
# 以下、HRFlow, LRFlow, HFFlow, DegFlow, SDFlowModel, 判別器クラスなどの定義が続きます
################################################################################

class HRFlow(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, n_levels=3, n_flow_steps=4):
        super(HRFlow, self).__init__()
        self.flow = Flow(in_channels, hidden_channels, n_levels, n_flow_steps)

    def forward(self, x=None, reverse=False, z=None):
        if not reverse:
            z, logdet = self.flow(x)
            z_c, z_h = torch.chunk(z, 2, dim=1)
            return z_c, z_h, logdet
        else:
            return self.flow(z, reverse=True)


class LRFlow(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, n_levels=3, n_flow_steps=4):
        super(LRFlow, self).__init__()
        self.flow = Flow(in_channels, hidden_channels, n_levels, n_flow_steps)

    def forward(self, x=None, reverse=False, z=None):
        if not reverse:
            z, logdet = self.flow(x)
            z_c, z_d = torch.chunk(z, 2, dim=1)
            return z_c, z_d, logdet
        else:
            return self.flow(z, reverse=True)


class HFFlow(nn.Module):
    def __init__(self, z_c_channels=32, z_h_channels=32, n_blocks=8):
        super(HFFlow, self).__init__()
        # RRDB を条件ネットワークとして使う
        self.condition_net = self._make_rrdb(n_blocks, in_channels=z_c_channels)
        self.flow = Flow(z_h_channels * 2, z_h_channels, 1, 2)

    def _make_rrdb(self, n_blocks, in_channels):
        layers = []
        for _ in range(n_blocks):
            layers.append(RRDB(in_channels))
        return nn.Sequential(*layers)

    def forward(self, z_h=None, z_c=None, reverse=False, temp=1.0):
        cond = self.condition_net(z_c)
        if not reverse:
            inp = torch.cat([z_h, cond], dim=1)
            z, logdet = self.flow(inp)
            return z, logdet
        else:
            batch = z_c.size(0)
            mean = torch.zeros((batch, z_h.size(1), z_h.size(2), z_h.size(3)), device=z_c.device)
            eps = torch.randn_like(mean) * temp
            z_prior = mean + eps
            inp = torch.cat([z_prior, cond], dim=1)
            return self.flow(inp, reverse=True)


class DegFlow(nn.Module):
    def __init__(self, z_c_channels=32, z_d_channels=32, n_blocks=4, n_components=16):
        super(DegFlow, self).__init__()
        self.condition_net = self._make_rrdb(n_blocks, in_channels=z_c_channels)
        self.means = nn.Parameter(torch.zeros(n_components, z_d_channels))
        self.log_vars = nn.Parameter(torch.zeros(n_components, z_d_channels))
        self.flow = Flow(z_d_channels * 2, z_d_channels, 1, 2)

    def _make_rrdb(self, n_blocks, in_channels):
        layers = []
        for _ in range(n_blocks):
            layers.append(RRDB(in_channels))
        return nn.Sequential(*layers)

    def forward(self, z_d=None, z_c=None, reverse=False, temp=1.0):
        cond = self.condition_net(z_c)
        if not reverse:
            inp = torch.cat([z_d, cond], dim=1)
            z, logdet = self.flow(inp)
            return z, logdet
        else:
            batch = z_c.size(0)
            comp_idx = torch.randint(0, self.means.size(0), (batch,), device=z_c.device)
            means = self.means[comp_idx]
            vars = torch.exp(self.log_vars[comp_idx])
            eps = torch.randn_like(means) * torch.sqrt(vars) * temp
            z_prior = means + eps
            inp = torch.cat([z_prior, cond], dim=1)
            return self.flow(inp, reverse=True)


class SDFlowModel(nn.Module):
    def __init__(self):
        super(SDFlowModel, self).__init__()
        self.hr_flow = HRFlow(in_channels=1, hidden_channels=64, n_levels=3, n_flow_steps=4)
        self.lr_flow = LRFlow(in_channels=1, hidden_channels=64, n_levels=3, n_flow_steps=4)
        self.hf_flow = HFFlow(z_c_channels=32, z_h_channels=32, n_blocks=8)
        self.deg_flow = DegFlow(z_c_channels=32, z_d_channels=32, n_blocks=4, n_components=16)

    def forward_hr(self, hr_img):
        z_c_hr, z_h, logdet_hr = self.hr_flow(hr_img)
        z_h_trans, logdet_hf = self.hf_flow(z_h, z_c_hr)
        return z_c_hr, z_h_trans, None, None, logdet_hr + logdet_hf

    def forward_lr(self, lr_img):
        z_c_lr, z_d, logdet_lr = self.lr_flow(lr_img)
        z_d_trans, logdet_deg = self.deg_flow(z_d, z_c_lr)
        return z_c_lr, z_d_trans, None, None, logdet_lr + logdet_deg

    def generate_sr(self, lr_img, temp=0.0):
        z_c_lr, _ = self.lr_flow(lr_img)
        z_h = self.hf_flow(None, z_c_lr, reverse=True, temp=temp)
        z = torch.cat([z_c_lr, z_h], dim=1)
        return self.hr_flow(None, reverse=True, z=z)

    def generate_ds(self, hr_img, temp=0.0):
        z_c_hr, _ = self.hr_flow(hr_img)
        z_d = self.deg_flow(None, z_c_hr, reverse=True, temp=temp)
        z = torch.cat([z_c_hr, z_d], dim=1)
        return self.lr_flow(None, reverse=True, z=z)


# ■■ 画像判別器 / 潜在判別器 の定義 ■■
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
