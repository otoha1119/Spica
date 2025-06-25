import torch
import torch.nn as nn
from sdflow import modules as m  # ActNorm, Inv1x1Conv2d, AffineInjector, AffineCoupling など

class Identity(nn.Module):
    """何もしないモジュール。条件付き情報がない場合や最終ステップで使用"""
    def forward(self, z, ldj, *args, reverse=False):
        return z, ldj

class FlowStep(nn.Module):
    """
    Flow の 1 つのステップ。
    1) ActNorm → 2) Permutation (Inv1x1Conv) → 3) AffineInjector (条件付き注入) → 4) AffineCoupling
    """
    def __init__(
        self,
        z_channels,
        hidden_layers,
        hidden_channels,
        permute='inv1x1',
        condition_channels=None,
        affine_split=0.5,
        with_actnorm=True,
        is_last=False
    ):
        super().__init__()
        self.is_last = is_last
        self.z_channels = z_channels

        # 1) ActNorm
        self.actnorm = m.ActNorm(z_channels) if with_actnorm else Identity()
        # 2) Permutation
        self.permute = m.Inv1x1Conv2d(z_channels) if permute == 'inv1x1' else Identity()

        # 3) AffineInjector (条件付き注入)
        if condition_channels is not None and z_channels > 0:
            self.affine_injector = m.AffineInjector(
                z_channels, hidden_layers, hidden_channels, condition_channels
            )
        else:
            self.affine_injector = Identity()

        # 4) AffineCoupling (条件付き)
        if not self.is_last and z_channels > 1 and condition_channels is not None:
            self.affine_coupling = m.AffineCoupling(
                z_channels,
                hidden_layers,
                hidden_channels,
                condition_channels,
                split_ratio=affine_split
            )
        else:
            self.affine_coupling = Identity()

    def forward(self, z, ldj, u=None, reverse=False):
        if not reverse:
            z, ldj = self.actnorm(z, ldj)
            z, ldj = self.permute(z, ldj)
            z, ldj = self.affine_injector(z, ldj, u)
            z, ldj = self.affine_coupling(z, ldj, u)
        else:
            z, ldj = self.affine_coupling(z, ldj, u, reverse=True)
            z, ldj = self.affine_injector(z, ldj, u, reverse=True)
            z, ldj = self.permute(z, ldj, reverse=True)
            z, ldj = self.actnorm(z, ldj, reverse=True)
        return z, ldj

class Flow(nn.Module):
    """
    FlowStep を n_flows 個つなげたシーケンスラッ퍼。
    順方向では (z, ldj) を返し、逆方向では z_out のみ返す。
    """
    def __init__(self, in_channels, hidden_channels, n_levels, n_flows):
        super().__init__()
        blocks = []
        z_channels = in_channels
        for i in range(n_flows):
            is_last = (i == n_flows - 1)
            blocks.append(
                FlowStep(
                    z_channels,
                    hidden_layers=2,
                    hidden_channels=hidden_channels,
                    permute='inv1x1',
                    condition_channels=None,
                    affine_split=0.5,
                    with_actnorm=True,
                    is_last=is_last
                )
            )
        self.flow = nn.ModuleList(blocks)

    def forward(self, x, reverse=False):
        batch_size = x.size(0)
        ldj = torch.zeros(batch_size, device=x.device)
        if not reverse:
            z = x
            for layer in self.flow:
                z, ldj = layer(z, ldj)
            return z, ldj
        else:
            z = x
            for layer in reversed(self.flow):
                z, ldj = layer(z, ldj, reverse=True)
            return z

class Inv1x1Conv2d(nn.Module):
    # ...（既存実装をそのまま使用）
    pass

class ZeroConv2d(nn.Module):
    # ...（既存実装をそのまま使用）
    pass
