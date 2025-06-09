# sdflow/models.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sdflow.flow_block import FlowStep, Flow   # FlowStep／Flow を使う
import sdflow.modules as m                      # ConditionalFlow, RRDB, gaussian_logp など


class HRFlow(nn.Module):
    """
    HRFlow: 高解像度画像用の Flow モジュール。
    ここではサンプルとして「in_channels→in_channels の FlowStep」を n_flows 回連ねる構成にしています。
    """
    def __init__(self, in_channels=1, hidden_channels=64, n_levels=3, n_flows=4):
        super().__init__()
        layers = []
        for _ in range(n_flows):
            layers.append(
                FlowStep(
                    z_channels=in_channels,
                    hidden_layers=2,
                    hidden_channels=hidden_channels,
                    permute='inv1x1',
                    condition_channels=None,
                    affine_split=0.5,
                    with_actnorm=True,
                    is_last=False
                )
            )
        self.flow = nn.ModuleList(layers)

    def forward(self, x=None, reverse=False, z=None):
        if not reverse:
            # 順方向
            batch = x.size(0)
            ldj = torch.zeros(batch, device=x.device)
            flow_input = x
            for layer in self.flow:
                flow_input, ldj = layer(flow_input, ldj)
            z_c, z_h = torch.chunk(flow_input, 2, dim=1)
            
            
            
            return z_c, z_h, ldj
        else:
            # 逆方向
            flow_input = z
            ldj = None 
            for layer in reversed(self.flow):
                flow_input, ldj = layer(flow_input, ldj, reverse=True)
            out, _ = torch.chunk(flow_input, 2, dim=1)
            return out


class ContentFlow(nn.Module):
    """
    ContentFlow: z_c 上の追加的な FlowStep（ここではシンプルに 1 層だけ）。
    """
    def __init__(self):
        super().__init__()
        self.flow = nn.Sequential(
            FlowStep(
                z_channels=1,
                hidden_layers=2,
                hidden_channels=64,
                permute='inv1x1',
                condition_channels=None,
                affine_split=0.5,
                with_actnorm=True,
                is_last=True
            )
        )

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            out, ldj = self.flow[0](z, ldj)
            return out, ldj
        else:
            out, ldj = self.flow[0](z, ldj, reverse=True)
            return out, ldj

    def requires_grad(self, mode: bool):
        for p in self.parameters():
            p.requires_grad_(mode)


class DegFlow(nn.Module):
    """
    DegFlow: 劣化潜在 z_d を ConditionalFlow で扱うモジュール。
    """
    def __init__(self, in_channels, cond_channels, n_gaussian):
        super().__init__()
        self.n_gaussian = n_gaussian
        self.means = nn.Parameter(torch.randn(n_gaussian))
        self.log_stds = nn.Parameter(torch.zeros(n_gaussian))
        self.deg_net = m.ConditionalFlow(
            in_channels=in_channels,
            condition_channels=cond_channels,
            is_squeeze=True,
            n_steps=8,
            learned_prior=False,
            compute_log_prob=False
        )

    def forward(self, z, ldj, u, tau=0.0, gaussian_id=None, reverse=False):
        if gaussian_id is None:
            gaussian_id = list(range(self.n_gaussian))
        else:
            gaussian_id = [gaussian_id] if isinstance(gaussian_id, int) else list(gaussian_id)

        if not reverse:
            b, c, h, w = z.shape
            z, ldj = self.deg_net(z, ldj, u, reverse=False)
            all_logps = torch.cat(
                [
                    m.gaussian_logp(
                        z,
                        self.means[i].expand_as(z),
                        self.log_stds[i].expand_as(z)
                    ).reshape(b, 1, -1)
                    for i in gaussian_id
                ],
                dim=1
            )
            mixture_all_logps = torch.logsumexp(all_logps, dim=1) - np.log(len(gaussian_id))
            ldj = ldj + mixture_all_logps.sum(dim=-1)
            return z, ldj

        else:
            b, _, h, w = u.shape
            if self.deg_net.is_squeeze:
                h, w = h // 2, w // 2
            c = self.deg_net.z_channels
            gaussian_idx = np.random.choice(len(gaussian_id), size=(b,))
            all_samples = torch.cat(
                [
                    m.gaussian_sample(
                        self.means[idx].expand(1, c, h, w),
                        self.log_stds[idx].expand(1, c, h, w),
                        tau
                    )
                    for idx in gaussian_idx
                ],
                dim=0
            )
            if ldj is not None:
                all_logps = torch.cat(
                    [
                        m.gaussian_logp(
                            all_samples,
                            self.means[i].expand_as(all_samples),
                            self.log_stds[i].expand_as(all_samples)
                        ).reshape(b, 1, -1)
                        for i in gaussian_id
                    ],
                    dim=1
                )
                mixture_all_logps = torch.logsumexp(all_logps, dim=1) - np.log(len(gaussian_id))
                ldj = ldj - mixture_all_logps.sum(dim=-1)
            z, ldj = self.deg_net(all_samples, ldj, u, reverse=True)
            return z, ldj
