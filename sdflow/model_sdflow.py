import torch
import torch.nn as nn
from sdflow.flow_block import FlowStep, Flow
from sdflow.modules import AffineCoupling, RRDB

class HRFlow(nn.Module):
    """HR 画像用の Glow ライクなフロー"""
    def __init__(self, in_channels=1, hidden_channels=64, n_levels=3, n_flows=4):
        super().__init__()
        self.flow = Flow(in_channels, hidden_channels, n_levels, n_flows)

    def forward(self, x, reverse=False, z=None):
        if not reverse:
            z_out, logdet = self.flow(x)
            z_c, z_h = torch.chunk(z_out, 2, dim=1)
            return z_c, z_h, logdet
        else:
            return self.flow(z, reverse=True)

class LRFlow(nn.Module):
    """LR 画像用の Glow ライクなフロー"""
    def __init__(self, in_channels=1, hidden_channels=64, n_levels=3, n_flows=4):
        super().__init__()
        self.flow = Flow(in_channels, hidden_channels, n_levels, n_flows)

    def forward(self, x, reverse=False, z=None):
        if not reverse:
            z_out, logdet = self.flow(x)
            z_c, z_d = torch.chunk(z_out, 2, dim=1)
            return z_c, z_d, logdet
        else:
            return self.flow(z, reverse=True)

class HFFlow(nn.Module):
    """高周波潜在用の条件付きフロー"""
    def __init__(self, z_c_channels=32, z_h_channels=32, n_blocks=8):
        super().__init__()
        self.condition_net = self._make_rrdb(n_blocks, z_c_channels)
        self.flow = Flow(z_h_channels * 2, z_h_channels, n_levels=1, n_flows=2)

    def _make_rrdb(self, n_blocks, in_channels):
        layers = []
        for _ in range(n_blocks):
            layers.append(RRDB(in_channels))
        return nn.Sequential(*layers)

    def forward(self, z_h, z_c, reverse=False, temp=1.0):
        cond = self.condition_net(z_c)
        if not reverse:
            inp = torch.cat([z_h, cond], dim=1)
            z_out, logdet = self.flow(inp)
            return z_out, logdet
        else:
            shape = (z_c.size(0), z_h.size(1), z_h.size(2), z_h.size(3))
            eps = torch.randn(shape, device=z_c.device) * temp
            inp = torch.cat([eps, cond], dim=1)
            z_rev = self.flow(inp, reverse=True)
            return z_rev

class DegFlow(nn.Module):
    """劣化潜在用の条件付きフロー"""
    def __init__(self, z_c_channels=32, z_d_channels=32, n_blocks=4, n_components=16):
        super().__init__()
        self.condition_net = self._make_rrdb(n_blocks, z_c_channels)
        self.means = nn.Parameter(torch.zeros(n_components, z_d_channels))
        self.log_vars = nn.Parameter(torch.zeros(n_components, z_d_channels))
        self.flow = Flow(z_d_channels * 2, z_d_channels, n_levels=1, n_flows=2)

    def _make_rrdb(self, n_blocks, in_channels):
        layers = []
        for _ in range(n_blocks):
            layers.append(RRDB(in_channels))
        return nn.Sequential(*layers)

    def forward(self, z_d, z_c, reverse=False, temp=1.0):
        cond = self.condition_net(z_c)
        if not reverse:
            inp = torch.cat([z_d, cond], dim=1)
            z_out, logdet = self.flow(inp)
            return z_out, logdet
        else:
            batch = z_c.size(0)
            comp_idx = torch.randint(0, self.means.size(0), (batch,), device=z_c.device)
            mu = self.means[comp_idx]
            sigma = torch.exp(0.5 * self.log_vars[comp_idx])
            eps = sigma * torch.randn_like(mu) * temp
            z_prior = mu + eps
            inp = torch.cat([z_prior, cond], dim=1)
            z_rev = self.flow(inp, reverse=True)
            return z_rev

class SDFlowModel(nn.Module):
    """SDFlow全体モデル: HR/LR フロー＋HF/Deg フローを組み合わせ"""
    def __init__(self, in_channels=1, hidden_channels=64,
                 n_levels=3, n_flows=4,
                 hf_blocks=8, deg_blocks=4, deg_mixture=16):
        super().__init__()
        self.hr_flow = HRFlow(in_channels, hidden_channels, n_levels, n_flows)
        self.lr_flow = LRFlow(in_channels, hidden_channels, n_levels, n_flows)
        self.hf_flow = HFFlow(z_c_channels=hidden_channels//2, z_h_channels=hidden_channels//2, n_blocks=hf_blocks)
        self.deg_flow = DegFlow(z_c_channels=hidden_channels//2, z_d_channels=hidden_channels//2, n_blocks=deg_blocks, n_components=deg_mixture)

    def encode_hr(self, hr_img):
        z_c, z_h, logdet1 = self.hr_flow(hr_img, reverse=False)
        z_h_trans, logdet2 = self.hf_flow(z_h, z_c, reverse=False)
        logdet = logdet1 + logdet2
        return z_c, z_h_trans, logdet

    def encode_lr(self, lr_img):
        z_c, z_d, logdet1 = self.lr_flow(lr_img, reverse=False)
        z_d_trans, logdet2 = self.deg_flow(z_d, z_c, reverse=False)
        logdet = logdet1 + logdet2
        return z_c, z_d_trans, logdet

    def generate_sr(self, lr_img, temp=0.0):
        z_c, _, _ = self.lr_flow(lr_img, reverse=False)
        z_h = self.hf_flow(None, z_c, reverse=True, temp=temp)
        z_comb = torch.cat([z_c, z_h], dim=1)
        sr = self.hr_flow(None, reverse=True, z=z_comb)
        return sr

    def generate_ds(self, hr_img, temp=0.0):
        z_c, _, _ = self.hr_flow(hr_img, reverse=False)
        z_d = self.deg_flow(None, z_c, reverse=True, temp=temp)
        z_comb = torch.cat([z_c, z_d], dim=1)
        ds = self.lr_flow(None, reverse=True, z=z_comb)
        return ds
