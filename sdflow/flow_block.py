import torch
import torch.nn as nn
from sdflow import modules as m  # ActNorm, Inv1x1Conv2d, AffineInjector, AffineCoupling など

class Identity(nn.Module):
    """何もしないモジュール。チャネル数が 1 以下のときや最終ステップ用に使う"""
    def forward(self, z, ldj, *args, reverse=False):
        return z, ldj


class FlowStep(nn.Module):
    """
    Flow の 1 つのステップ。
    1) ActNorm → 2) Permutation(Inv1x1Conv) → 3) [AffineInjector] → 4) [AffineCoupling]
    z_channels <= 1 のときは AffineCoupling を Identity に置き換えて 0 チャネル conv を回避する。
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

        # 1) ActNorm 部分
        if with_actnorm:
            self.actnorm = m.ActNorm(z_channels)
        else:
            self.actnorm = Identity()

        # 2) Permutation 部分 (Inv1x1Conv or Identity)
        if permute == 'inv1x1':
            self.permute = m.Inv1x1Conv2d(z_channels)
        else:
            self.permute = Identity()

        # 3) AffineInjector 部分 (条件付き注入)。condition_channels があり、かつ z_channels > 0 のときのみ有効
        if (condition_channels is not None) and (z_channels > 0):
            self.affine_injector = m.AffineInjector(
                z_channels, hidden_layers, hidden_channels, condition_channels
            )
        else:
            self.affine_injector = Identity()

        # 4) AffineCoupling 部分
        #    - is_last=True の最後ステップ、または z_channels <= 1 のときは Identity
        if (not self.is_last) and (z_channels > 1):
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
        """
        z:    (B, z_channels, H, W)
        ldj:  (B,) — ログデターミナントを累積するテンソル
        u:    条件テンソル (任意)
        reverse: 逆フローを行うか否か
        """
        if not reverse:
            # 順方向
            z, ldj = self.actnorm(z, ldj)
            z, ldj = self.permute(z, ldj)
            if not isinstance(self.affine_injector, Identity):
                z, ldj = self.affine_injector(z, ldj, u)
            z, ldj = self.affine_coupling(z, ldj, u)
        else:
            # 逆方向順序
            z, ldj = self.affine_coupling(z, ldj, u, reverse=True)
            if not isinstance(self.affine_injector, Identity):
                z, ldj = self.affine_injector(z, ldj, u, reverse=True)
            z, ldj = self.permute(z, ldj, reverse=True)
            z, ldj = self.actnorm(z, ldj, reverse=True)

        return z, ldj


class Flow(nn.Module):
    """
    FlowStep を n_flows 個つなげたシーケンスラッパー。
    順方向では (z_out, ldj) を返し、逆方向では z_out のみ返す。
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
            # FlowStep はチャネル数を変えないため z_channels はそのまま

        self.flow = nn.Sequential(*blocks)

    def forward(self, x, reverse=False):
        """
        x: (B, in_channels, H, W)
        reverse=False → (z, ldj)、reverse=True → 画像再構築 z_out
        """
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
            return z  # 逆方向では ldj を返さず、画像を返す


# model_sdflow.py が期待する名前をエクスポート
FlowStep = FlowStep
Flow = Flow