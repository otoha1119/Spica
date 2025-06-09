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
    1) ActNorm → 2) Permutation (Inv1x1Conv) → 3) [AffineInjector] → 4) [AffineCoupling]
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
            # 逆方向の順序
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

        self.flow = nn.ModuleList(blocks)

    def forward(self, x, reverse=False):
        """
        x: (B, in_channels, H, W)
        reverse=False → (z, ldj)、reverse=True → 再構築画像 z_out
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

            return z


class Inv1x1Conv2d(nn.Module):
    """
    1x1 の逆畳み込みによるチャネル入れ替え (Inv1x1Conv)。
    順方向では重み行列をそのまま掛け、逆方向では逆行列を掛ける。
    チャネル数が 1 のときはスカラー逆数を計算して対応する。
    """
    def __init__(self, in_channel):
        super().__init__()

        weight = m.ops.randn(in_channel, in_channel).cpu().numpy()
        import numpy.linalg as la
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        self.register_buffer("w_p", torch.from_numpy(w_p.copy()))
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(torch.from_numpy(w_s)))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(torch.from_numpy(w_l))
        self.w_s = nn.Parameter(torch.log(torch.abs(torch.from_numpy(w_s))))
        self.w_u = nn.Parameter(torch.from_numpy(w_u))

    def calc_weight(self):
        """
        LU 分解された成分から重みテンソルを再構築して返す。
        形状は (C, C, 1, 1)。
        """
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, z, ldj, reverse=False):
        h, w = z.shape[2:]
        weight = self.calc_weight()   # shape: (C, C, 1, 1)
        C = weight.size(0)

        if not reverse:
            # 順方向：そのまま畳み込み
            z = F.conv2d(z, weight)
            ldj = ldj + h * w * torch.sum(self.w_s)
        else:
            # 逆方向：チャンネル数 C が 1 のときはスカラーの逆数を使う
            if C == 1:
                # weight は shape=(1,1,1,1) → スカラー w
                w_scalar = weight.view(1, 1)          # 形を (1,1) の 2 次元テンソルに変換
                inv_scalar = w_scalar.reciprocal()     # 1×1 行列の逆行列相当
                inv_weight = inv_scalar.view(1, 1, 1, 1)
            else:
                # C×C 行列を逆行列にする
                w_mat = weight.view(C, C)             # → shape=(C, C)
                inv_mat = w_mat.inverse()             # → shape=(C, C)
                inv_weight = inv_mat.view(C, C, 1, 1)

            z = F.conv2d(z, inv_weight)
            if ldj is not None:
                ldj = ldj - h * w * torch.sum(self.w_s)

        return z, ldj


class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, logscale_factor=3.):
        super().__init__()
        self.logscale_factor = logscale_factor

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.logs = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, z):
        z = F.pad(z, [1, 1, 1, 1], value=0)
        z = self.conv(z)
        z = z * torch.exp(self.logs * self.logscale_factor)
        return z
