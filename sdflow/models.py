import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sdflow.flow_block import FlowBlock
from sdflow import modules as m
from sdflow import ops

class HRFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        # --- 変更点：split_size=3 を「正」とし、以降のすべてのチャネル数計算の基準とする ---
        self.blocks.append(FlowBlock(1, 3, 64, 16, is_squeeze=True, is_split=False, permute='inv1x1'))
        self.blocks.append(FlowBlock(4, 3, 64, 16, is_squeeze=True, is_split=True, permute='inv1x1', split_size=4))
    
    def forward(self, z, ldj, tau=0, reverse=False):
        if not reverse:
            z_h = None
            for block in self.blocks:
                if hasattr(block, 'is_split') and block.is_split:
                    z, z_h, ldj = block(z, ldj, tau)
                else:
                    z, ldj = block(z, ldj, tau)
            return z, z_h, ldj
        else:
            for block in self.blocks[::-1]:
                z, ldj = block(z, ldj, tau, reverse=True)
            return z, ldj

class ContentFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(FlowBlock(1, 3, 64, 16, is_squeeze=True, is_expand=True, is_split=False, permute='inv1x1'))
    
    def forward(self, z, ldj, reverse=False):
        if not reverse:
            for block in self.blocks:
                z, ldj = block(z, ldj)
            return z, ldj
        else:
            for block in self.blocks[::-1]:
                z, ldj = block(z, ldj, reverse=True)
            return z, ldj
    
    def requires_grad(self, mode):
        for p in self.parameters():
            p.requires_grad_(mode)

class DegFlow(nn.Module):
    def __init__(self, in_channels, cond_channels, n_gaussian):
        super().__init__()
        self.n_gaussian = n_gaussian
        self.means = nn.Parameter(torch.randn(n_gaussian))
        self.log_stds = nn.Parameter(torch.zeros(n_gaussian))
        self.deg_net = m.ConditionalFlow(in_channels, cond_channels, is_squeeze=True, n_steps=8, learned_prior=False, compute_log_prob=False)
    
    def forward(self, z, ldj, u, tau=0., gaussian_id=None, reverse=False):
        if gaussian_id is None: gaussian_id = list(range(self.n_gaussian))
        else: gaussian_id = [gaussian_id] if isinstance(gaussian_id, int) else list(gaussian_id)

        if not reverse:
            b, c, h, w = z.shape
            z, ldj = self.deg_net(z, ldj, u, reverse=False)
            all_logps = torch.cat([m.gaussian_logp(z, self.means[i].expand_as(z), self.log_stds[i].expand_as(z)).reshape(b, 1, -1) for i in gaussian_id], dim=1)
            mixture_all_logps = torch.logsumexp(all_logps, dim=1) - np.log(len(gaussian_id))
            ldj = ldj + mixture_all_logps.sum(dim=-1)
            return z, ldj
        else:
            if ldj is None and u is not None:
                ldj = torch.zeros(u.size(0), device=u.device)
            b, _, h, w = u.shape
            if self.deg_net.is_squeeze: h, w = h // 2, w // 2
            c = self.deg_net.z_channels
            gaussian_idx = np.random.choice(len(gaussian_id), size=(b))
            all_samples_z = torch.cat([m.gaussian_sample(self.means[idx].expand(1, c, h, w), self.log_stds[idx].expand(1, c, h, w), tau) for idx in gaussian_idx], dim=0)

            if ldj is not None:
                all_logps = torch.cat([m.gaussian_logp(all_samples_z, self.means[i].expand_as(all_samples_z), self.log_stds[i].expand_as(all_samples_z)).reshape(b, 1, -1) for i in gaussian_id], dim=1)
                mixture_all_logps = torch.logsumexp(all_logps, dim=1) - np.log(len(gaussian_id))
                ldj = ldj - mixture_all_logps.sum(dim=-1)
            
            z, ldj = self.deg_net(all_samples_z, ldj, u, reverse=True)
            return z, ldj