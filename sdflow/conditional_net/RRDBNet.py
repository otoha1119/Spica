# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE


# sdflow/conditional_net/RRDBNet.py

import torch
import torch.nn as nn

# condtional_net を直接参照している行を修正
from . import module_util as mutil
from torch.nn import Conv2d
from torch.nn import LeakyReLU
from torch.nn import BatchNorm2d
from torch.nn import Upsample
from torch.nn import Sequential
from torch.nn import Module

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, growth_channels, kernel_size=3, padding=1)
        self.lrelu1 = LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = Conv2d(in_channels + growth_channels, growth_channels, kernel_size=3, padding=1)
        self.lrelu2 = LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = Conv2d(in_channels + 2 * growth_channels, growth_channels, kernel_size=3, padding=1)
        self.lrelu3 = LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv4 = Conv2d(in_channels + 3 * growth_channels, growth_channels, kernel_size=3, padding=1)
        self.lrelu4 = LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv5 = Conv2d(in_channels + 4 * growth_channels, in_channels, kernel_size=3, padding=1)
        self.scale = 0.2  # 係数

    def forward(self, x):
        x1 = self.lrelu1(self.conv1(x))
        x2 = self.lrelu2(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu3(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu4(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x + x5 * self.scale

class RRDB(Module):
    """
    Residual in Residual Dense Block for conditional network
    """
    def __init__(self, in_channels, growth_channels=32, num_blocks=3):
        super(RRDB, self).__init__()
        self.rdb_1 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb_2 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb_3 = ResidualDenseBlock(in_channels, growth_channels)
        self.scale = 0.2

    def forward(self, x):
        out = self.rdb_1(x)
        out = self.rdb_2(out)
        out = self.rdb_3(out)
        return x + out * self.scale



class RRDBNet(nn.Module):
    def __init__(self, gc=32, scale=4):
        nf = 64
        nb = 8
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.scale = scale

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.scale >= 8:
            self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.scale >= 16:
            self.upconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.scale >= 32:
            self.upconv5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, get_steps=False):
        fea_x = self.conv_first(x)
        fea = fea_x.clone()

        block_idxs = [1, 3, 5, 7] # Index of RRDB encoder to concat features
        block_results = {}

        for idx, m in enumerate(self.RRDB_trunk.children()):
            fea = m(fea)
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea

        trunk = self.trunk_conv(fea)

        last_lr_fea = fea_x + trunk

        fea_up2 = self.upconv1(F.interpolate(last_lr_fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(fea_up2)

        fea_up4 = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(fea_up4)

        fea_up8 = None
        fea_up16 = None
        fea_up32 = None

        if self.scale >= 8:
            fea_up8 = self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(fea_up8)
        if self.scale >= 16:
            fea_up16 = self.upconv4(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(fea_up16)
        if self.scale >= 32:
            fea_up32 = self.upconv5(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(fea_up32)

        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        results = {'last_lr_fea': last_lr_fea,
                   'fea_up1': last_lr_fea,
                   'fea_up2': fea_up2,
                   'fea_up4': fea_up4,
                   'fea_up8': fea_up8,
                   'fea_up16': fea_up16,
                   'fea_up32': fea_up32,
                   'out': out}

        fea_up0 = True
        fea_up_1 = False
        if fea_up0:
            results['fea_up0'] = F.interpolate(last_lr_fea, scale_factor=1/2, mode='bilinear',
                                               align_corners=False, recompute_scale_factor=True)

        if fea_up_1:
            results['fea_up_1'] = F.interpolate(last_lr_fea, scale_factor=1/4, mode='bilinear',
                                                align_corners=False, recompute_scale_factor=True)

        if get_steps:
            for k, v in block_results.items():
                results[k] = v
            return results
        else:
            return out


if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = RRDBNet(scale=4).to(device)
    summary(encoder, (3, 160, 160))