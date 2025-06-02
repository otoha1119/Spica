# ===============================================
# sdflow/model_sdflow.py
# ===============================================

import torch
import torch.nn as nn

# FlowStep や Flow を使う
from sdflow.flow_block import FlowStep, Flow

# RRDB を使う (sdflow/modules.py 内に定義されているもの)
from sdflow.modules import RRDB

# HRFlow の定義またはインポートを忘れないでください。
# もしすでに別のファイルに HRFlow を定義済みならば、以下のようにインポートしてください。
# from sdflow.hrflow import HRFlow
# あるいは、このファイル内に HRFlow クラス定義を追加してください。

# たとえば、HRFlow を同ファイルに置く場合は、
# class HRFlow(nn.Module):
#     def __init__(self, in_channels, hidden_channels, n_levels, n_flows):
#         super().__init__()
#         # ...（元々の実装をここに貼り付け）
#     def forward(self, x=None, reverse=False, z=None):
#         # ...（元々の実装をここに貼り付け）
#         pass
# として定義してください。


class HFFlow(nn.Module):
    """
    高周波潜在 (z_h) を扱う Flow。
    - 順方向: (z_h, z_c) → RRDB(cond) → Flow → (z_out, logdet)
    - 逆方向: z_c → RRDB(cond) → 乱数 eps → Flow 逆 → z_h 再構築
    """
    def __init__(self, z_c_channels=1, z_h_channels=1, n_blocks=4):
        super().__init__()
        # 条件ネットワークとして RRDB を n_blocks 個積む
        self.condition_net = self._make_rrdb(n_blocks, z_c_channels)

        # Flow 部分：入力チャンネル数 = z_h_channels + z_c_channels、出力チャンネル数 = z_h_channels
        self.flow = Flow(z_h_channels + z_c_channels, z_h_channels, n_levels=1, n_flows=2)

        # 逆方向時に生成する z_h のチャンネル数を保持
        self.z_h_channels = z_h_channels

    def _make_rrdb(self, n_blocks, in_channels):
        """
        n_blocks 個分の RRDBBlock を積み重ねた nn.Sequential を返す。
        Args:
            n_blocks (int): RRDB を何層重ねるか
            in_channels (int): RRDB の入力チャネル数（= z_c のチャネル数）
        Returns:
            nn.Sequential: RRDB ブロックを並べたモジュール
        """
        layers = []
        for _ in range(n_blocks):
            layers.append(RRDB(in_channels))
        return nn.Sequential(*layers)

    def forward(self, z_h=None, z_c=None, reverse=False, temp=1.0):
        """
        Args:
            z_h (Tensor or None): 順方向時は高周波潜在テンソル (B, z_h_channels, H, W)。
                                 逆方向時は None（サンプリングで生成する）。
            z_c (Tensor): 条件となる潜在 z_c (B, z_c_channels, H, W)
            reverse (bool): 逆方向を実行するなら True（サンプリング）、それ以外は False（順伝播）。
            temp (float): 逆方向サンプリング時の温度
        Returns:
            - 順方向: (z_out, logdet)
            - 逆方向: z_rev （再構築された高周波潜在テンソル）
        """
        # (A) RRDB で z_c を処理し、条件特徴 cond を生成
        cond = self.condition_net(z_c)

        if not reverse:
            # (B) 順方向: z_h と cond をチャンネル方向に結合し、Flow を通す
            inp = torch.cat([z_h, cond], dim=1)  # (B, z_h_channels+z_c_channels, H, W)
            z_out, logdet = self.flow(inp)
            return z_out, logdet

        else:
            # (C) 逆方向: z_h が None のため、乱数 eps でサンプリング
            batch = z_c.size(0)
            H, W = z_c.size(2), z_c.size(3)
            shape = (batch, self.z_h_channels, H, W)
            eps = torch.randn(shape, device=z_c.device) * temp

            inp = torch.cat([eps, cond], dim=1)  # (B, z_h_channels+z_c_channels, H, W)
            z_rev = self.flow(inp, reverse=True)
            return z_rev


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
            # 順方向: x → Flow → (z_out, logdet)。z_d はゼロにする
            z_out, logdet = self.flow(x)
            z_c = z_out
            z_d = torch.zeros_like(z_out)
            return z_c, z_d, logdet
        else:
            # z_comb をそのまま逆伝播して画像を復元
            return self.flow(z, reverse=True)


class DegFlow(nn.Module):
    """
    劣化潜在 (z_d) 用の条件付き Flow。
    - 順方向: z_d, z_c → RRDB(cond) → Flow → (z_d_trans, logdet)
    - 逆方向: z_c からガウス混合分布によるサンプリング → Flow 逆 → z_d 再構築
    """
    def __init__(self, z_c_channels=1, z_d_channels=1, n_blocks=4, n_components=8):
        super().__init__()
        self.condition_net = self._make_rrdb(n_blocks, z_c_channels)
        # ガウス混合の平均・分散をパラメータとして定義
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
            # 順方向: z_d と cond を結合して Flow を通す
            inp = torch.cat([z_d, cond], dim=1)
            z_out, logdet = self.flow(inp)
            return z_out, logdet
        else:
            # 逆方向: ガウス混合に従い z_prior をサンプリング
            batch = z_c.size(0)
            comp_idx = torch.randint(0, self.means.size(0), (batch,), device=z_c.device)
            mu = self.means[comp_idx]                   # (B, z_d_channels)
            sigma = torch.exp(0.5 * self.log_vars[comp_idx])
            eps = sigma * torch.randn_like(mu) * temp
            z_prior = mu + eps                          # (B, z_d_channels)

            # z_prior を (B, z_d_channels, H, W) に拡張
            H, W = z_c.size(2), z_c.size(3)
            z_prior = z_prior.view(batch, -1, 1, 1).expand(batch, z_prior.size(1), H, W)

            inp = torch.cat([z_prior, cond], dim=1)     # (B, z_d_channels+z_c_channels, H, W)
            z_rev = self.flow(inp, reverse=True)
            return z_rev


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
        deg_mixture=8
    ):
        super().__init__()
        # HRFlow の実装が別ファイルにある想定。定義済みでなければ同ファイルに貼り付けてください。
        self.hr_flow = HRFlow(in_channels, hidden_channels, n_levels, n_flows)
        self.lr_flow = LRFlow(in_channels, hidden_channels, n_levels, n_flows)
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
        """
        HR 画像のエンコード（順方向）。  
        1. hr_img → HRFlow → (z_c, z_h, logdet1)  
        2. z_h, z_c → HFFlow → (z_h_trans, logdet2)  
        合計 logdet = logdet1 + logdet2 を返す。
        Returns:
            z_c      : 潜在 z_c (B, z_c_channels, H', W')
            z_h_trans: 高周波変換後の潜在 z_h (B, z_h_channels, H', W')
            total_ldj: Flow の順方向ログデターミナントの合計 (B,)
        """
        z_c, z_h, logdet1 = self.hr_flow(hr_img, reverse=False)
        z_h_trans, logdet2 = self.hf_flow(z_h, z_c, reverse=False)
        total_ldj = logdet1 + logdet2
        return z_c, z_h_trans, total_ldj

    def encode_lr(self, lr_img):
        """
        LR 画像のエンコード（順方向）。  
        1. lr_img → LRFlow → (z_c, z_d, logdet1)  
        2. z_d, z_c → DegFlow → (z_d_trans, logdet2)  
        合計 logdet = logdet1 + logdet2 を返す。
        Returns:
            z_c       : 潜在 z_c (B, z_c_channels, H', W')
            z_d_trans : 劣化潜在 z_d の変換後 (B, z_d_channels, H', W')
            total_ldj : Flow の順方向ログデターミナントの合計 (B,)
        """
        z_c, z_d, logdet1 = self.lr_flow(lr_img, reverse=False)
        z_d_trans, logdet2 = self.deg_flow(z_d, z_c, reverse=False)
        total_ldj = logdet1 + logdet2
        return z_c, z_d_trans, total_ldj

    def generate_sr(self, lr_img, temp=0.0):
        """
        LR 画像から SR 画像を逆方向サンプリングで生成。  
        1. lr_img → LRFlow(順) → z_c  
        2. z_c → HFFlow(逆, temp) → z_h  
        3. z_comb = cat([z_c, z_h]) → HRFlow(逆) → sr_image  
        Returns:
            sr_image: 生成された超解像画像 (B, 1, H, W)
        """
        z_c, _, _ = self.lr_flow(lr_img, reverse=False)
        z_h = self.hf_flow(None, z_c, reverse=True, temp=temp)
        z_comb = torch.cat([z_c, z_h], dim=1)
        sr = self.hr_flow(None, reverse=True, z=z_comb)
        return sr

    def generate_ds(self, hr_img, temp=0.0):
        """
        HR 画像から DS 画像を逆方向サンプリングで生成。  
        1. hr_img → HRFlow(順) → z_c  
        2. z_c → DegFlow(逆, temp) → z_d  
        3. z_comb = cat([z_c, z_d]) → LRFlow(逆) → ds_image  
        Returns:
            ds_image: 生成されたダウンサンプル画像 (B, 1, H, W)
        """
        z_c, _, _ = self.hr_flow(hr_img, reverse=False)
        z_d = self.deg_flow(None, z_c, reverse=True, temp=temp)
        z_comb = torch.cat([z_c, z_d], dim=1)
        ds = self.lr_flow(None, reverse=True, z=z_comb)
        return ds
