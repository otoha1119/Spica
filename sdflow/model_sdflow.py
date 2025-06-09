import torch.nn.functional as F
import torch
import torch.nn as nn

# FlowStep や Flow を使う
from sdflow.flow_block import FlowStep, Flow
from sdflow.models import HRFlow    # 追加
# RRDB を使う (sdflow/modules.py 内に定義されているもの)
from sdflow.modules import RRDB



class HFFlow(nn.Module):
    """
    高周波潜在 (z_h) を扱う Flow。
    - 順方向: (z_h, z_c) → RRDB(cond) → Flow → (z_out, logdet)
    - 逆方向: z_c → RRDB(cond) → 乱数 eps → Flow 逆 → z_h 再構築
    """
    def __init__(self, z_c_channels=1, z_h_channels=1, n_blocks=4):
        super().__init__()
        self.condition_net = self._make_rrdb(n_blocks, z_c_channels)
        self.flow = Flow(z_h_channels + z_c_channels, z_h_channels, n_levels=1, n_flows=2)
        self.z_h_channels = z_h_channels

    def _make_rrdb(self, n_blocks, in_channels):
        layers = []
        for _ in range(n_blocks):
            layers.append(RRDB(in_channels))
        return nn.Sequential(*layers)

    def forward(self, z_h=None, z_c=None, reverse=False, temp=1.0):
        cond = self.condition_net(z_c)
        if not reverse:
            inp = torch.cat([z_h, cond], dim=1)
            z_out, logdet = self.flow(inp)
            return z_out, logdet
        else:
            batch = z_c.size(0)
            H, W = z_c.size(2), z_c.size(3)
            shape = (batch, self.z_h_channels, H, W)
            eps = torch.randn(shape, device=z_c.device) * temp
            inp = torch.cat([eps, cond], dim=1)
            z_rev = self.flow(inp, reverse=True)
            z_h_reconstructed, _ = torch.split(z_rev, [self.z_h_channels, cond.size(1)], dim=1)
            return z_h_reconstructed

class LRFlow(nn.Module):
    """
    低解像度画像 (LR) 用の Flow ラッパー。
    - 順方向: x → Flow → (z_out, logdet) → z_c = z_out, z_d = all-zero
    - 逆方向: z_comb → Flow 逆 → 画像再構築
    """
    def __init__(self, in_channels=1, hidden_channels=64, n_levels=3, n_flows=4):
        super().__init__()
        self.flow = Flow(in_channels, hidden_channels, n_levels, n_flows)

    def forward(self, x=None, reverse=False, z=None):
        if not reverse:
            # 順方向
            
            
            flow_output = self.flow(x)
            
            

            z_out, logdet = flow_output
            z_c, z_d = torch.chunk(z_out, 2, dim=1)
            return z_c, z_d, logdet
        else:
            # 逆方向
            out = self.flow(z, reverse=True)
            out, _ = torch.chunk(out, 2, dim=1)
            return out


class DegFlow(nn.Module):
    """
    劣化潜在 (z_d) 用の条件付き Flow。
    - 順方向: z_d, z_c → RRDB(cond) → Flow → (z_d_trans, logdet)
    - 逆方向: z_c からガウス混合分布によるサンプリング → Flow 逆 → z_d 再構築
    """
    def __init__(self, z_c_channels=1, z_d_channels=1, n_blocks=4, n_components=8):
        super().__init__()
        self.condition_net = self._make_rrdb(n_blocks, z_c_channels)
        self.means = nn.Parameter(torch.zeros(n_components, z_d_channels))
        self.log_vars = nn.Parameter(torch.zeros(n_components, z_d_channels))
        self.flow = Flow(z_d_channels + z_c_channels, z_d_channels, n_levels=1, n_flows=2)

    def _make_rrdb(self, n_blocks, in_channels):
        layers = []
        for _ in range(n_blocks):
            layers.append(RRDB(in_channels))
        return nn.Sequential(*layers)

    def forward(self, z_d=None, z_c=None, reverse=False, temp=1.0):
        cond = self.condition_net(z_c)
        if not reverse:
            inp = torch.cat([z_d, cond], dim=1)
            z_out, logdet = self.flow(inp)
            return z_out, logdet
        else:
            batch = z_c.size(0)
            comp_idx = torch.randint(0, self.means.size(0), (batch,), device=z_c.device)
            mu = self.means[comp_idx]
            sigma = torch.exp(0.5 * self.log_vars[comp_idx])
            eps = sigma * torch.randn_like(mu) * temp
            z_prior = mu + eps
            H, W = z_c.size(2), z_c.size(3)
            z_prior = z_prior.view(batch, -1, 1, 1).expand(batch, z_prior.size(1), H, W)
            inp = torch.cat([z_prior, cond], dim=1)
            z_rev = self.flow(inp, reverse=True)
            z_d_reconstructed, _ = torch.split(z_rev, [z_prior.size(1), cond.size(1)], dim=1)
            return z_d_reconstructed


class SDFlowModel(nn.Module):
    """
    SDFlow の全体モデル:
      HRFlow + LRFlow + HFFlow + DegFlow を組み合わせ、以下のメソッドを提供する。
      - encode_hr   : HR 画像を Flow でエンコードし、(z_c, z_h_trans, total_logdet) を返す
      - encode_lr   : LR 画像を Flow でエンコードし、(z_c, z_d_trans, total_logdet) を返す
      - generate_sr : LR 画像から SR 画像を生成（逆方向サンプリング）
      - generate_ds : HR 画像から DS 画像を生成（逆方向サンプリング）
    """
    def __init__(
        self,
        in_channels=1,
        hidden_channels=64,
        n_levels=3,
        n_flows=4,
        hf_blocks=4,
        deg_blocks=4,
        deg_mixture=8,
        scale=4
    ):
        super().__init__()
        self.scale = scale
        # hr_flowとlr_flowはz_c, z_h/z_dを結合したものを入力とするためin_channels * 2で初期化
        self.hr_flow = HRFlow(in_channels * 2, hidden_channels, n_levels, n_flows)
        self.lr_flow = LRFlow(in_channels * 2, hidden_channels, n_levels, n_flows)
        # 他の*_channelsはin_channels(1)のまま
        self.hf_flow = HFFlow(
            z_c_channels=in_channels,
            z_h_channels=in_channels,
            n_blocks=hf_blocks
        )
        self.deg_flow = DegFlow(
            z_c_channels=in_channels,
            z_d_channels=in_channels,
            n_blocks=deg_blocks,
            n_components=deg_mixture
        )

    def encode_hr(self, hr_img):
        b, c, h, w = hr_img.shape
        padded_hr = torch.cat([hr_img, torch.zeros(b, c, h, w, device=hr_img.device)], dim=1)
        z_c, z_h, logdet1 = self.hr_flow(padded_hr, reverse=False)
        
        z_h_trans, logdet2 = self.hf_flow(z_h, z_c, reverse=False)
        total_ldj = logdet1 + logdet2

        
        return z_c, z_h_trans, total_ldj
        # ...
    def encode_lr(self, lr_img):
        b, c, h, w = lr_img.shape
        padded_lr = torch.cat([lr_img, torch.zeros(b, c, h, w, device=lr_img.device)], dim=1)
        z_c, z_d, logdet1 = self.lr_flow(padded_lr, reverse=False)
        
        z_d_trans, logdet2 = self.deg_flow(z_d, z_c, reverse=False)
        total_ldj = logdet1 + logdet2
        return z_c, z_d_trans, total_ldj
    
    
    def generate_sr(self, lr_img, temp=0.0):
        z_c, _, _ = self.lr_flow(lr_img, reverse=False)
        z_h = self.hf_flow(None, z_c, reverse=True, temp=temp)
        z_comb = torch.cat([z_c, z_h], dim=1)
        sr = self.hr_flow(None, reverse=True, z=z_comb)
        
        # --- 修正：生成したsrを正しいサイズにアップサンプリング ---
        return F.interpolate(sr, scale_factor=self.scale, mode='bilinear', align_corners=False)

    def generate_ds(self, hr_img, temp=0.0):
        z_c, _, _ = self.hr_flow(hr_img, reverse=False)
        z_d = self.deg_flow(None, z_c, reverse=True, temp=temp)
        z_comb = torch.cat([z_c, z_d], dim=1)
        ds = self.lr_flow(None, reverse=True, z=z_comb)
        
        # --- 修正：生成したdsを正しいサイズにダウンサンプリング ---
        return F.interpolate(ds, scale_factor=1/self.scale, mode='bilinear', align_corners=False)