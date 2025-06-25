# sdflow/models.py (1チャンネル特化版)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sdflow.flow_block import FlowBlock
from sdflow import modules as m

class HRFlow(nn.Module):
    # in_channels, n_flows, scale を引数で受け取り、1chに特化させず柔軟な作りにする
    def __init__(self, in_channels=1, n_flows=4, hidden_channels=64, scale=2):
        super().__init__()
        
        # スケールに応じたマルチレベル構造を定義
        n_levels = int(np.log2(scale))
        self.blocks = nn.ModuleList()
        
        current_channels = in_channels
        
        # 最初のレベル
        self.blocks.append(FlowBlock(z_channels=current_channels, hidden_layers=3, hidden_channels=hidden_channels, 
                                     n_steps=n_flows, is_squeeze=True, is_split=True, permute='inv1x1', split_size=in_channels))
        # squeezeで4倍になった後、z_hとして分離されるin_channels分を引く
        current_channels = current_channels * 4 - in_channels
        
        
        # 2番目以降のレベル (scale=4以上の場合に実行される)
        for i in range(n_levels - 1):
            # 論文の設計通り、最後の分割ブロックにするかどうかを判断
            is_last_split = (i == n_levels - 2)
            self.blocks.append(FlowBlock(z_channels=current_channels, hidden_layers=3, hidden_channels=hidden_channels, 
                                         n_steps=n_flows, is_squeeze=True, is_split=is_last_split, permute='inv1x1', split_size=in_channels))
            if is_last_split:
                current_channels = current_channels // 2 # split後は半分になる
            current_channels = current_channels * 4

    def forward(self, z, ldj, tau=0., u_hf=None, reverse=False):
        if not reverse:
            # 順方向の処理
            z_h_list = []
            for block in self.blocks:
                if block.is_split:
                    z, z_h, ldj = block(z, ldj, tau)
                    z_h_list.append(z_h)
                else:
                    z, ldj = block(z, ldj, tau)
            # 論文では複数のz_hを扱うが、ここでは最後のものだけを返す
            return z, z_h_list[-1] if z_h_list else None, ldj
        else:
            # 逆方向の処理
            if u_hf is not None:
                z = [z, u_hf]
            for block in reversed(self.blocks):
                z, ldj = block(z, ldj, tau, reverse=True)
            return z, ldj

class ContentFlow(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1チャンネル入力、n_flows=16 に固定
        in_channels = 1
        n_flows = 16
        hidden_channels = 64
        
        self.blocks = nn.ModuleList()
        self.blocks.append(FlowBlock(
            z_channels=in_channels,
            hidden_layers=3,
            hidden_channels=hidden_channels,
            n_steps=n_flows,
            is_squeeze=True,
            is_expand=True,
            is_split=False,
            permute='inv1x1'
        ))

    def forward(self, z, ldj, reverse=False):
        # このforwardはuを受け取らないので、シグネチャを合わせる
        if not reverse:
            for block in self.blocks:
                z, ldj = block(z, ldj)
            return z, ldj
        else:
            for block in self.blocks[::-1]:
                z, ldj = block(z, ldj, reverse=True)
            return z, ldj


class DegFlow(nn.Module):
    def __init__(self, in_channels, cond_channels, n_gaussian):
        super().__init__()

        self.n_gaussian = n_gaussian
        # チャンネル次元を持たない、シンプルなパラメータとして定義
        self.means = nn.Parameter(torch.randn(n_gaussian))
        self.log_stds = nn.Parameter(torch.zeros(n_gaussian))

        # ConditionalFlowに確率計算をさせない設定で初期化
        self.deg_net = m.ConditionalFlow(in_channels, cond_channels, is_squeeze=True, n_steps=8, learned_prior=False, compute_log_prob=False)

    def forward(self, z, u, ldj=None, tau=0., gaussian_id=None, reverse=False):
        if ldj is None:
            ldj = torch.zeros(z.size(0) if z is not None else u.size(0), device=u.device)
            
        if gaussian_id is None: gaussian_id = list(range(self.n_gaussian))
        else: gaussian_id = [gaussian_id] if isinstance(gaussian_id, int) else list(gaussian_id)

        if not reverse:
            b, c, h, w = z.shape
            z, ldj = self.deg_net(z, ldj, u, reverse=False)
            
            # .expand_as(z) を使い、meansとlog_stdsをデータzの形に合わせてから計算
            all_logps = torch.cat([m.gaussian_logp(z, self.means[i].expand_as(z), self.log_stds[i].expand_as(z)).reshape(b, 1, -1) for i in gaussian_id], dim=1)
            
            mixture_all_logps = torch.logsumexp(all_logps, dim=1) - np.log(len(gaussian_id))
            ldj = ldj + mixture_all_logps.sum(dim=-1)

        else:
            b, _, h, w = u.shape
            if self.deg_net.is_squeeze: h, w = h // 2, w // 2
            c = self.deg_net.z_channels
            gaussian_idx = np.random.choice(len(gaussian_id), size=(b))
            
            # .expand() を使い、サンプルを生成
            all_samples = torch.cat([m.gaussian_sample(self.means[idx].expand(1, c, h, w), self.log_stds[idx].expand(1, c, h, w), tau) for idx in gaussian_idx], dim=0)

            if ldj is not None:
                # こちらも .expand_as() で形を合わせる
                all_logps = torch.cat([m.gaussian_logp(all_samples, self.means[i].expand_as(all_samples), self.log_stds[i].expand_as(all_samples)).reshape(b, 1, -1) for i in gaussian_id], dim=1)
                mixture_all_logps = torch.logsumexp(all_logps, dim=1) - np.log(len(gaussian_id))
                ldj = ldj - mixture_all_logps.sum(dim=-1)
            
            z, ldj = self.deg_net(all_samples, ldj, u, reverse=True)

        return z, ldj