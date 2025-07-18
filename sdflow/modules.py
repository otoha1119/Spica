from math import log, pi, sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from scipy import linalg as la

# --- 変更点1: プロジェクトの構成に合わせたインポートパス修正 ---
# 元の "import conditional_net.module_util as mutil" 等から変更
from sdflow.conditional_net import module_util as mutil
from sdflow.conditional_net.RRDBNet import RRDB
from sdflow import ops
# --- 変更ここまで ---

from einops import rearrange, reduce


def gaussian_logp(x, mean, log_std):
    # return -0.5 * log(2. * pi) - log_std - 0.5 * ((x - mean) ** 2) / (torch.exp(2. * log_std))
    c = log(2 * pi)
    if mean is None and log_std is None:
        return -0.5 * (x ** 2 + c)
    elif log_std is None:
        return -0.5 * ((x - mean) ** 2 + c)
    else:
        return -0.5 * (log_std * 2. + ((x - mean) ** 2) / (torch.exp(log_std * 2.) + 1e-6) + c)


def gaussian_sample(mean, log_std, temperature=1.):
    if temperature < 1e-6:
        return mean
    eps = torch.normal(mean=torch.zeros_like(mean).detach(), std=torch.ones_like(log_std).detach() * temperature)
    return mean + (torch.exp(log_std) + 1e-6) * eps


class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, res_scale=0.1):
        super().__init__()

        self.res_scale = res_scale
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, hidden_channels, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_channels, in_channels, 3)
        )
    
    def forward(self, x):
        return x + self.conv(x) * self.res_scale


class FeatureModulation(nn.Module):
    def __init__(self, in_channels, mf_channels):
        super().__init__()

        self.in_channels = in_channels
        self.mf_channels = mf_channels
        self.log_scale = nn.Parameter(torch.zeros(1,))
        self.affine = nn.Conv2d(mf_channels, in_channels * 2, 1)
    
    def forward(self, feat, mfeat):
        s, b = torch.chunk(self.affine(mfeat), 2, dim=1)
        scale = torch.exp(self.log_scale)
        s = torch.tanh(s / (scale + 1e-6)) * scale
        feat = feat * s + b

        return feat


class FeatureModulatedResBlock(nn.Module):
    def __init__(self, in_channels, mf_channels, hidden_channels, res_scale=0.1):
        super().__init__()

        self.res_scale = res_scale
        self.fm1 = FeatureModulation(in_channels, mf_channels)
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_channels, hidden_channels, 3), nn.LeakyReLU(0.2, inplace=True))
        self.fm2 = FeatureModulation(hidden_channels, mf_channels)
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(hidden_channels, in_channels, 3))
    
    def forward(self, feat, mfeat):
        mf = self.fm1(feat, mfeat)
        mf = self.conv1(mf)
        mf = self.fm2(mf, mfeat)
        mf = self.conv2(mf)

        return feat + mf * self.res_scale


class CheckboardSqueeze(nn.Module):
    def forward(self, z, ldj, reverse=False):
        b, c, h, w = z.shape
        if not reverse:
            assert(h % 2 == 0)
            assert(w % 2 == 0)
            z = z.reshape(b, c, h // 2, 2, w // 2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
            z = z.reshape(b, 4 * c, h // 2, w // 2)
        else:
            assert(c % 4 == 0)
            z = z.reshape(b, c // 4, 2, 2, h, w)
            z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
            z = z.reshape(b, c // 4, h * 2, w * 2)

        return z, ldj


class CheckboardExpand(nn.Module):
    def forward(self, z, ldj, reverse=False):
        b, c, h, w = z.shape
        if not reverse:
            assert(c % 4 == 0)
            z = z.reshape(b, c // 4, 2, 2, h, w)
            z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
            z = z.reshape(b, c // 4, h * 2, w * 2)
        else:
            assert(h % 2 == 0)
            assert(w % 2 == 0)
            z = z.reshape(b, c, h // 2, 2, w // 2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
            z = z.reshape(b, 4 * c, h // 2, w // 2)

        return z, ldj


class ActNorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.prev_log_scale = self.log_scale.data

    def initialize(self, z):
        with torch.no_grad():
            # --- 変更点2: 分散学習(multi-GPU)用のコードを削除し、単一GPUで動作するように修正 ---
            mean = torch.mean(z, dim=[0, 2, 3], keepdim=True)
            var = torch.mean((z - mean) ** 2, dim=[0, 2, 3], keepdim=True)
            
            # 非常に小さい分散値に対する安全策は維持
            var = torch.where(var < 1e-4, torch.ones_like(var).detach(), var)
            
            log_std = torch.log(1. / (torch.sqrt(var) + 1e-6))
            
            self.loc.data.copy_(-mean)
            self.log_scale.data.copy_(log_std)
            # --- 変更ここまで ---

    def forward(self, z, ldj, reverse=False):
        h, w = z.shape[2:]

        if not reverse:
            if self.initialized.item() == 0:
                self.initialize(z)
                self.initialized.fill_(1)
            
            z = torch.exp(self.log_scale) * (z + self.loc)

            if ldj is not None:
                ldj = ldj + h * w * torch.sum(self.log_scale)
        else:
            z = z * torch.exp(-self.log_scale) - self.loc

            if ldj is not None:
                ldj = ldj - h * w * torch.sum(self.log_scale)

        return z, ldj

class Inv1x1Conv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p.copy())
        w_l = torch.from_numpy(w_l.copy())
        w_s = torch.from_numpy(w_s.copy())
        w_u = torch.from_numpy(w_u.copy())

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def forward(self, z, ldj, reverse=False):
        h, w = z.shape[2:]
        if not reverse:
            weight = self.calc_weight()

            z = F.conv2d(z, weight)
            ldj = ldj + h * w * torch.sum(self.w_s)
        
        else:
            weight = self.calc_weight()
            z = F.conv2d(z, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
            if ldj is not None:
                ldj = ldj - h * w * torch.sum(self.w_s)

        return z, ldj

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)


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


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, do_actnorm=True, weight_std=0.05):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias=(not do_actnorm))
        # init weight with std
        self.conv2d.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.conv2d.bias.data.zero_()
        else:
            self.actnorm = ActNorm(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, x):
        x = F.pad(x, [self.padding] * 4, mode='reflect')
        x = self.conv2d(x)
        if self.do_actnorm:
            x, _ = self.actnorm(x, None)
        return x


class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, hidden_layers=1, kernel_hidden=1, init='xavier', for_flow=True, do_actnorm=False):
        super(FCN, self).__init__()
        self.conv1 = Conv2d(in_channels, hidden_channels, kernel_size=3, do_actnorm=do_actnorm)
        self.conv2 = Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_hidden, do_actnorm=do_actnorm)
        self.conv3 = ZeroConv2d(hidden_channels, out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

        if for_flow:
            mutil.initialize_weights(self.conv3, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class ConditionNet(nn.Module):
    # --- 変更点3: `in_channes`のタイポを修正し、引数で渡せるように変更 ---
    def __init__(self, in_channels, out_channels=64, nb=8):
        super().__init__()

        self.in_conv = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_channels, 64, 3))
        # --- 変更ここまで ---
        self.trunk = nn.Sequential(*[RRDB(64, 32) for _ in range(nb)])
        self.trans = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3))
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, out_channels, 3)
        )

        mutil.initialize_weights_xavier([self.in_conv, self.trunk, self.out_conv], 0.1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.trans(self.trunk(x)) + x
        x = self.out_conv(x)

        return x


class AffineCoupling(nn.Module):
    def __init__(self, z_channels, hidden_layers, hidden_channels, condition_channels=None, split_ratio=0.5):
        super().__init__()

        self.conditional_channels = condition_channels
        self.channels_a = int(z_channels * split_ratio)
        self.channels_b = z_channels - self.channels_a
        if condition_channels is not None:
            self.net = FCN(self.channels_a + condition_channels, self.channels_b * 2, hidden_channels, hidden_layers)
        else:
            self.net = FCN(self.channels_a, self.channels_b * 2, hidden_channels, hidden_layers)

    def forward(self, z, ldj, u=None, reverse=False):
        if self.conditional_channels is None and u is not None:
            raise RuntimeError('Conditional is None but u is not None')
            
        if not reverse:
            za, zb = torch.split_with_sizes(z, [self.channels_a, self.channels_b], dim=1)
            za_u = ops.cat_feature(za, u) if self.conditional_channels is not None else za

            scale, shift = ops.split_feature(self.net(za_u), 'cross')
            scale = 0.318 * torch.atan(2 * scale)
        
            zb = (zb + shift) * torch.exp(scale)

            z = ops.cat_feature(za, zb)
            ldj = ldj + scale.sum(dim=[1, 2, 3])

        else:
            za, zb = torch.split_with_sizes(z, [self.channels_a, self.channels_b], dim=1)
            za_u = ops.cat_feature(za, u) if self.conditional_channels is not None else za
            
            scale, shift = ops.split_feature(self.net(za_u), 'cross')
            scale = 0.318 * torch.atan(2 * scale)
            zb = zb * torch.exp(-scale) - shift
            if ldj is not None:
                ldj = ldj - scale.sum(dim=[1, 2, 3])

            z = ops.cat_feature(za, zb)

        return z, ldj


class AffineInjector(nn.Module):
    def __init__(self, z_channels, hidden_layers, hidden_channels, condition_channels):
        super().__init__()

        self.conditional_channels = condition_channels
        self.net = FCN(condition_channels, z_channels * 2, hidden_channels, hidden_layers)

    def forward(self, z, ldj, u, reverse=False):
        if not reverse:
            scale, shift = ops.split_feature(self.net(u), 'cross')
            scale = 0.318 * torch.atan(2 * scale)
        
            z = (z + shift) * torch.exp(scale)

            ldj = ldj + scale.sum(dim=[1, 2, 3])

        else:
            scale, shift = ops.split_feature(self.net(u), 'cross')
            scale = 0.318 * torch.atan(2 * scale)
            z = z * torch.exp(-scale) - shift
            if ldj is not None:
                ldj = ldj - scale.sum(dim=[1, 2, 3])

        return z, ldj


class ConditionalFlow(nn.Module):
    def __init__(self, in_channels, condition_channels, is_squeeze=True, n_steps=16, n_resblock=4, learned_prior=True, compute_log_prob=True):
        super().__init__()
        self.is_squeeze = is_squeeze
        self.learned_prior = learned_prior
        self.compute_log_prob = compute_log_prob
        factor = 4 if self.is_squeeze else 1
        
        if self.is_squeeze:
            # self.cond_net から nn.PixelUnshuffle(2) と、それに伴うチャネル数調整を削除
            self.cond_net = nn.Sequential(
                ConditionNet(condition_channels, nb=n_resblock),
                # nn.PixelUnshuffle(2), # この行を削除またはコメントアウト
                # nn.Conv2d(64 * factor, 64, 1), # factorを削除
                nn.Conv2d(64, 64, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.squeeze = CheckboardSqueeze()
            self.squeeze_transition = TransitionBlock(in_channels * factor)
        else:
            # --- 変更点4: `ConditionNet`に`condition_channels`を渡す ---
            self.cond_net = ConditionNet(condition_channels, nb=n_resblock)
            # --- 変更ここまで ---
            self.squeeze = Identity()
            self.squeeze_transition = Identity()

        self.additional_flow_steps = nn.ModuleList()
        for step_idx in range(n_steps):
            self.additional_flow_steps.append(FlowStep(in_channels * factor, 3, 64, 'inv1x1', 64, affine_split=0.5, is_last=(step_idx == n_steps - 1)))
        
        if self.learned_prior:
            self.prior = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.2, True),
                ZeroConv2d(64, in_channels * factor * 2)
            )
        self.z_channels = in_channels * factor
    
    def forward(self, z, ldj, u, tau=0, reverse=False):
        u = self.cond_net(u)

        if not reverse:
            z, ldj = self.squeeze(z, ldj)
            z, ldj = self.squeeze_transition(z, ldj)
            for step in self.additional_flow_steps:
                z, ldj = step(z, ldj, u)
            
            if self.compute_log_prob:
                if self.learned_prior:
                    mean, log_std = ops.split_feature(self.prior(u), 'cross')
                    log_std = 0.318 * torch.atan(2 * log_std)
                else:
                    mean = torch.zeros_like(z).detach()
                    log_std = torch.zeros_like(z).detach()
                ldj = ldj + gaussian_logp(z, mean, log_std).sum(dim=[1, 2, 3])
        else:
            if self.compute_log_prob:
                if self.learned_prior:
                    mean, log_std = ops.split_feature(self.prior(u), 'cross')
                    log_std = 0.318 * torch.atan(2 * log_std)
                else:
                    b, c, h, w = u.shape
                    mean = torch.zeros(b, self.z_channels, h, w, device=u.get_device())
                    log_std = torch.zeros(b, self.z_channels, h, w, device=u.get_device())

                z = gaussian_sample(mean, log_std, tau)
                if ldj is not None: ldj = ldj - gaussian_logp(z, mean, log_std).sum(dim=[1, 2, 3])

            assert z is not None
            for step in self.additional_flow_steps[::-1]:
                z, ldj = step(z, ldj, u, reverse=True)
            z, ldj = self.squeeze_transition(z, ldj, reverse=True)
            z, ldj = self.squeeze(z, ldj, reverse=True)

        return z, ldj


class SplitFlow(nn.Module):
    def __init__(self, in_channels, split_size=None, n_resblock=3):
        super().__init__()
        if split_size is None:
            self.za_channels = in_channels // 2
        else:
            self.za_channels = split_size
        self.zb_channels = in_channels - self.za_channels
        self.conditional_flow = ConditionalFlow(self.zb_channels, self.za_channels, False, learned_prior=False, n_steps=8, n_resblock=n_resblock)
    
    def forward(self, z, ldj, tau, reverse=False):
        # --- 変更点5: models.pyの動作と合わせるための必須修正 ---
        if not reverse:
            za, zb = torch.split(z, [self.za_channels, self.zb_channels], dim=1)
            _, ldj = self.conditional_flow(zb, ldj, za.detach(), tau)

            # zb(次のブロックに渡すz) と za(分離されたz_h) を両方返す
            return zb, za, ldj
        else:
            if not isinstance(z, (list, tuple)):
                raise ValueError("逆伝播時には [z_c, z_h] のリストで渡してください")
            
            zb, za = z # models.pyから [z_c, z_h] の形で渡されることを想定
            z = torch.cat([za, zb], dim=1)
            
            # 元のコードではldjがNoneのままconditional_flowに渡されていたため修正
            if ldj is None:
                ldj = torch.zeros(z.size(0), device=z.device)
                
            # conditional_flowの逆伝播は尤度計算のみ（zは生成しない）
            # ここでは何もしないのが正しい
            
            return z, ldj
        # --- 変更ここまで ---


class Identity(nn.Module):
    def forward(self, z, ldj, *args, reverse=False):
        return z, ldj


class TransitionStep(nn.Module):
    def __init__(self, in_channels, with_actnorm=True):
        super().__init__()

        if with_actnorm:
            self.actnorm = ActNorm(in_channels)
        else:
            self.actnorm = Identity()
        self.inv_conv = Inv1x1Conv2d(in_channels)
    
    def forward(self, z, ldj, reverse=False):
        z = z.to(self.inv_conv.w_l.device)

        if not reverse:
            z, ldj = self.actnorm(z, ldj)
            z, ldj = self.inv_conv(z, ldj)
        else:
            z, ldj = self.inv_conv(z, ldj, reverse=True)
            z, ldj = self.actnorm(z, ldj, reverse=True)
        
        return z, ldj


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, steps=2, with_actnorm=True):
        super().__init__()
        self.steps = steps

        self.transitions = nn.ModuleList()
        for _ in range(self.steps):
            self.transitions.append(TransitionStep(in_channels, with_actnorm))
            pass
    
    def forward(self, z, ldj, reverse=False):
        if not reverse:
            for transition in self.transitions:
                z, ldj = transition(z, ldj)
        else:
            for transition in self.transitions[::-1]:
                z, ldj = transition(z, ldj, reverse=True)
        
        return z, ldj


class FlowStep(nn.Module):
    def __init__(self, z_channels, hidden_layers, hidden_channels, permute='inv1x1', condition_channels=None, affine_split=0.5, with_actnrom=True, is_last=False):
        super().__init__()

        self.is_last = is_last

        if with_actnrom:
            self.actnorm = ActNorm(z_channels)
        else:
            self.actnorm = Identity()
        if permute == 'inv1x1':
            self.permute = Inv1x1Conv2d(z_channels)
        else:
            self.permute = Identity()
        
        # --- 変更点: 元コードではuがない場合にエラーが出るため修正 ---
        self.affine_injector = None
        if condition_channels is not None:
            self.affine_injector = AffineInjector(z_channels, hidden_layers, hidden_channels, condition_channels)
        
        if not self.is_last:
            self.affine_coulping = AffineCoupling(z_channels, hidden_layers, hidden_channels, condition_channels, split_ratio=affine_split)
        else:
            self.affine_coulping = Identity()

    def forward(self, z, ldj, u=None, reverse=False):
        if not reverse:
            z, ldj = self.actnorm(z, ldj)
            z, ldj = self.permute(z, ldj)
            if u is not None and self.affine_injector is not None:
                z, ldj = self.affine_injector(z, ldj, u)
            z, ldj = self.affine_coulping(z, ldj, u)
        else:
            z, ldj = self.affine_coulping(z, ldj, u, reverse=True)
            if u is not None and self.affine_injector is not None:
                z, ldj = self.affine_injector(z, ldj, u, reverse=True)
            z, ldj = self.permute(z, ldj, reverse=True)
            z, ldj = self.actnorm(z, ldj, reverse=True)
        
        return z, ldj


class LRImageEncoderFM(nn.Module):
    # --- 変更点6: `in_channels`を引数で渡せるように変更 ---
    def __init__(self, in_channels, out_channels=3, nc=64, n_res_block=16, is_downscale=False, with_uncertainty=False):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, nc, 3),
            nn.LeakyReLU(0.2, True)
        )
        # --- 変更ここまで ---

        self.trunk = nn.ModuleList()
        for _ in range(n_res_block):
            self.trunk.append(FeatureModulatedResBlock(nc, nc, nc))

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nc, nc, 3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nc, out_channels, 1)
        )

        mf_conv = list()
        for _ in range(8):
            mf_conv.extend([nn.ReflectionPad2d(1), nn.Conv2d(nc, nc, 3), nn.LeakyReLU(0.2, inplace=True)])
        self.mf_conv = nn.Sequential(*mf_conv)

        self.with_uncertainty = with_uncertainty
        if self.with_uncertainty:
            self.uncertain_conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(nc, nc, 3),
                nn.LeakyReLU(0.2, inplace=True),
                ZeroConv2d(nc, out_channels)
            )
    
    def forward(self, x):
        x = self.in_conv(x)
        mf = self.mf_conv(x)
        xm = x.clone()
        for t in self.trunk:
            xm = t(xm, mf)

        x = x + xm
        x = self.out_conv(x)

        if self.with_uncertainty:
            log_std = self.uncertain_conv(mf)
            return x.tanh(), log_std.clamp(-6, 3)
        return x