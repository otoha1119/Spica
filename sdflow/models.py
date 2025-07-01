import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 変更点: flow_block と modules のインポートパスを想定 ---
from sdflow.flow_block import FlowBlock
from sdflow import modules as m


class HRFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        # --- 変更点1: FlowBlockの入力チャネル数を 3 -> 1 に変更 ---
        self.blocks.append(FlowBlock(1, 3, 64, 16, is_squeeze=True, is_split=False, permute='inv1x1'))

        # --- 変更点2: 2番目のブロックの入力チャネル数を 12 -> 4 に変更 ---
        # 1ch入力がSqueezeされて4chになったことを受けるため。split_size=3 は論文再現のため維持。
        self.blocks.append(FlowBlock(4, 3, 64, 16, is_squeeze=True, is_split=True, permute='inv1x1', split_size=3))
    
    def forward(self, z, ldj, tau=0, reverse=False):
        # --- 変更点3: is_split=True のブロックの出力を正しく処理 ---
        # この修正は、チャネル数変更に伴い、is_split=Trueのブロックが
        # 意図通りに動作するために必須となります。
        if not reverse:
            z_h = None # 分割された潜在変数を格納する変数
            for block in self.blocks:
                # is_splitがTrueの場合、戻り値が3つになる
                if hasattr(block, 'is_split') and block.is_split:
                    z, z_h, ldj = block(z, ldj, tau)
                else:
                    z, ldj = block(z, ldj, tau)
            # 論文のz_cに相当するzと、z_hを返す
            return z, z_h, ldj
        else:
            for block in self.blocks[::-1]:
                z, ldj = block(z, ldj, tau, reverse=True)
            return z, ldj


class ContentFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        # --- 変更点4: FlowBlockの入力チャネル数を 3 -> 1 に変更 ---
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
    # --- 変更点なし ---
    # このクラス定義自体は、入力チャネル数に依存しないため変更は不要です。
    # 学習スクリプトでインスタンス化する際に正しいチャネル数を渡します。
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
            return z, ldj # 戻り値を統一
        else:
            # 元のコードではldjがNoneの場合にエラーが出る可能性があったため、安全策を追加
            if ldj is None and u is not None:
                ldj = torch.zeros(u.size(0), device=u.device)
            b, _, h, w = u.shape
            if self.deg_net.is_squeeze: h, w = h // 2, w // 2
            c = self.deg_net.z_channels
            gaussian_idx = np.random.choice(len(gaussian_id), size=(b))
            
            # 元のコードではzが未定義のまま使われる可能性があったため修正
            all_samples_z = torch.cat([m.gaussian_sample(self.means[idx].expand(1, c, h, w), self.log_stds[idx].expand(1, c, h, w), tau) for idx in gaussian_idx], dim=0)

            if ldj is not None:
                # all_samplesを渡すべきところをz(未定義)になっていた点を修正
                all_logps = torch.cat([m.gaussian_logp(all_samples_z, self.means[i].expand_as(all_samples_z), self.log_stds[i].expand_as(all_samples_z)).reshape(b, 1, -1) for i in gaussian_id], dim=1)
                mixture_all_logps = torch.logsumexp(all_logps, dim=1) - np.log(len(gaussian_id))
                ldj = ldj - mixture_all_logps.sum(dim=-1)
            
            z, ldj = self.deg_net(all_samples_z, ldj, u, reverse=True)
            return z, ldj