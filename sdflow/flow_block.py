import torch
import torch.nn as nn
from sdflow import modules as m

class FlowBlock(nn.Module):
    def __init__(self, z_channels, hidden_layers, hidden_channels, n_steps, permute='inv1x1', condition_channels=None, is_squeeze=True, squzze_type='checkboard',
                 is_expand=False, expand_type='checkboard', is_split=False, split_size=None, affine_split=0.5, with_actnorm=True):
        super().__init__()

        self.is_squeeze = is_squeeze
        self.is_expand = is_expand
        self.is_split = is_split

        assert squzze_type == expand_type
        self.squeeze_transition = None
        self.expand_transition = None
        if is_squeeze:
            if squzze_type == 'checkboard':
                self.squeeze = m.CheckboardSqueeze()
                self.squeeze_transition = m.TransitionBlock(z_channels * 4, with_actnorm=with_actnorm)
            elif squzze_type == 'haar':
                self.squeeze = m.HaarWaveletSqueeze(z_channels)
            else:
                raise NotImplemented('Not implemented squeeze type')
        if is_expand:
            if expand_type == 'checkboard':
                self.expand = m.CheckboardExpand()
                self.expand_transition = m.TransitionBlock(z_channels * 4 if self.is_squeeze else z_channels, with_actnorm=with_actnorm)
            elif expand_type == 'haar':
                self.expand = m.HaarWaveletExpand(z_channels if self.is_squeeze else z_channels // 4)
            else:
                raise NotImplemented('Not implemented expand type')

        self.steps = nn.ModuleList()
        if is_squeeze: z_channels = z_channels * 4
        for _ in range(n_steps):
            self.steps.append(m.FlowStep(z_channels, hidden_layers, hidden_channels, permute, condition_channels, affine_split=affine_split, with_actnrom=with_actnorm))
        if is_expand: z_channels = z_channels // 4
        if is_split:
            self.split = m.SplitFlow(z_channels, split_size, n_resblock=8)
    
    def forward(self, z, ldj, tau=0, u=None, reverse=False):
        if not reverse:
            if self.is_squeeze:
                z, ldj = self.squeeze(z, ldj)
                if self.squeeze_transition:
                    z, ldj = self.squeeze_transition(z, ldj)

            for step in self.steps:
                z, ldj = step(z, ldj, u)
            
            if self.is_expand:
                if self.expand_transition:
                    z, ldj = self.expand_transition(z, ldj)
                z, ldj = self.expand(z, ldj)
            
            if self.is_split:
                z, ldj = self.split(z, ldj, tau)

            return z, ldj

        else:
            if self.is_split:
                z, ldj = self.split(z, ldj, tau, reverse=True)
            
            if self.is_expand:
                z, ldj = self.expand(z, ldj, reverse=True)
                if self.expand_transition:
                    z, ldj = self.expand_transition(z, ldj, reverse=True)
            
            for step in self.steps[::-1]:
                z, ldj = step(z, ldj, u, reverse=True)
            
            if self.is_squeeze:
                if self.squeeze_transition:
                    z, ldj = self.squeeze_transition(z, ldj, reverse=True)
                z, ldj = self.squeeze(z, ldj, reverse=True)

            return z, ldj

# エイリアス定義: SDFlow のモデル側が期待する FlowStep と Flow をここで定義
FlowStep = FlowBlock

class Flow(nn.Module):
    """
    単純に複数の FlowBlock を階層的に適用するラッパークラス
    実装は必要に応じて拡張する
    """
    def __init__(self, in_channels, hidden_channels, n_levels, n_flows):
        super().__init__()
        # ここでは簡易的に、単一レベルに n_flows 個の FlowBlock を適用する構成とする
        blocks = []
        z_channels = in_channels
        for _ in range(n_flows):
            blocks.append(FlowBlock(z_channels, hidden_layers=2, hidden_channels=hidden_channels, n_steps=1, permute='inv1x1', condition_channels=None, is_squeeze=False, is_expand=False, is_split=False))
        self.flow = nn.Sequential(*blocks)

    def forward(self, x, reverse=False):
        ldj = torch.zeros(x.size(0), device=x.device)
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
