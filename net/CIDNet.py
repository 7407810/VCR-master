import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin


class VCFStage(nn.Module):
    def __init__(self, channels, mask_ratio=1/3, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.mask_ratio = mask_ratio
        self.eps = eps

        self.i_norm = nn.InstanceNorm2d(channels, affine=True)
        self.hv_norm = nn.InstanceNorm2d(channels, affine=True)

        self.fuse_i = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GELU()
        )
        self.fuse_hv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GELU()
        )

    def _covariance(self, x):
        b, c, h, w = x.shape
        n = h * w
        x = x.flatten(2)
        x = x - x.mean(dim=2, keepdim=True)
        cov = torch.bmm(x, x.transpose(1, 2)) / max(n, 1)
        return cov

    def _build_upper_mask(self, cov):
        b, c, _ = cov.shape
        device = cov.device

        upper_idx = torch.triu_indices(c, c, offset=1, device=device)
        upper_vals = cov[:, upper_idx[0], upper_idx[1]]
        num_pairs = upper_vals.shape[1]
        k = max(1, int(num_pairs * self.mask_ratio))

        topk_idx = upper_vals.topk(k=k, dim=1, largest=True).indices

        mask = cov.new_zeros((b, c, c))
        for bi in range(b):
            chosen = topk_idx[bi]
            mask[bi, upper_idx[0, chosen], upper_idx[1, chosen]] = 1.0

        return mask

    def _channel_gate_from_mask(self, mask):
        mask_sym = mask + mask.transpose(1, 2)
        score = mask_sym.sum(dim=2)
        score = score / (score.max(dim=1, keepdim=True)[0] + self.eps)
        gate = 1.0 - score
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        return gate

    def forward(self, fi, fhv):
        fi_n = self.i_norm(fi)
        fhv_n = self.hv_norm(fhv)

        d_i = self._covariance(fi_n)
        d_hv = self._covariance(fhv_n)

        mu = 0.5 * (d_i + d_hv)
        cov = 0.5 * ((d_i - mu).pow(2) + (d_hv - mu).pow(2))

        mask = self._build_upper_mask(cov)

        l_vcf = ((d_i.abs() * mask).sum(dim=(1, 2)) +
                 (d_hv.abs() * mask).sum(dim=(1, 2))).mean()

        gate = self._channel_gate_from_mask(mask)

        fi_filtered = fi_n * gate
        fhv_filtered = fhv_n * gate

        fi_out = self.fuse_i(torch.cat([fi, fi_filtered], dim=1))
        fhv_out = self.fuse_hv(torch.cat([fhv, fhv_filtered], dim=1))

        return fi_out, fhv_out, l_vcf


class GBPool(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.cat([max_pool, avg_pool], dim=1)


class TCEBranch(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.pool = GBPool()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv(attn)
        attn = self.bn(attn)
        attn = self.sigmoid(attn)
        return x * attn


class TCEStage(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.branch1 = TCEBranch(kernel_size)
        self.branch2 = TCEBranch(kernel_size)
        self.branch3 = TCEBranch(kernel_size)

    def forward(self, x):
        x1 = x.permute(0, 2, 1, 3).contiguous()
        x1 = self.branch1(x1)
        x1 = x1.permute(0, 2, 1, 3).contiguous()

        x2 = x.permute(0, 3, 1, 2).contiguous()
        x2 = self.branch2(x2)
        x2 = x2.permute(0, 2, 3, 1).contiguous()

        x3 = self.branch3(x)

        out = (x1 + x2 + x3) / 3.0
        return out + x


class CAAFront(nn.Module):
    def __init__(self, channels, mask_ratio=1/3, tce_kernel_size=7):
        super().__init__()
        self.vcf = VCFStage(channels, mask_ratio=mask_ratio)
        self.tce_i = TCEStage(kernel_size=tce_kernel_size)
        self.tce_hv = TCEStage(kernel_size=tce_kernel_size)

    def forward(self, fi, fhv):
        fi, fhv, l_vcf = self.vcf(fi, fhv)
        fi = self.tce_i(fi)
        fhv = self.tce_hv(fhv)
        return fi, fhv, l_vcf

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        channels=[36, 36, 72, 144],
        heads=[1, 2, 4, 8],
        norm=False,
        mask_ratio=1/3,
        tce_kernel_size=7,
    ):
        super(CIDNet, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # -----------------------------
        # Shallow stems
        # -----------------------------
        # HV branch only takes 2 chroma channels
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(2, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False)
        )

        # Front-end CAA = VCF + TCE
        self.front_caa = CAAFront(
            channels=ch1,
            mask_ratio=mask_ratio,
            tce_kernel_size=tce_kernel_size,
        )

        # -----------------------------
        # HV encoder / decoder
        # -----------------------------
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # -----------------------------
        # I encoder / decoder
        # -----------------------------
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False)
        )

        # -----------------------------
        # Cross-branch interaction
        # -----------------------------
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        # -----------------------------
        # CDA feature heads
        # pred side: internal enhanced HV feature -> color feature space
        # gt side  : GT HV map -> the same color feature space
        # use 2*ch1 to mimic paper's 2C notation
        # -----------------------------
        self.cda_dim = ch1 * 2

        self.cda_pred_head = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, self.cda_dim, 3, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(self.cda_dim, self.cda_dim, 3, stride=1, padding=0, bias=False),
        )

        self.cda_gt_head = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(2, self.cda_dim, 3, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(self.cda_dim, self.cda_dim, 3, stride=1, padding=0, bias=False),
        )

        self.trans = RGB_HVI()
        self.last_lvcf = None

    def forward(self, x, return_aux=False):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x).to(dtypes)

        # split HVI -> HV + I
        hv = hvi[:, :2, :, :]   # [B, 2, H, W]
        i = hvi[:, 2:3, :, :]   # [B, 1, H, W]

        # shallow projection
        i_enc0 = self.IE_block0(i)      # [B, ch1, H, W]
        hv_0 = self.HVE_block0(hv)      # [B, ch1, H, W]

        # front-end CAA
        i_enc0, hv_0, l_vcf = self.front_caa(i_enc0, hv_0)
        self.last_lvcf = l_vcf

        # encoder level 1
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc1 = self.IE_block1(i_enc0)
        hv_1 = self.HVE_block1(hv_0)

        # encoder level 2
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)

        i_jump1 = i_enc2
        hv_jump1 = hv_2

        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        # encoder level 3
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)

        i_jump2 = i_enc3
        hv_jump2 = hv_3

        i_enc3 = self.IE_block3(i_enc3)
        hv_3 = self.HVE_block3(hv_3)

        # bottleneck
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

        # decoder level 3
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, i_jump2)

        i_dec3 = self.I_LCA5(i_dec3, hv_3)
        hv_3 = self.HV_LCA5(hv_3, i_dec3)

        # decoder level 2
        hv_2 = self.HVD_block2(hv_3, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, i_jump1)

        i_dec2 = self.I_LCA6(i_dec2, hv_2)
        hv_2 = self.HV_LCA6(hv_2, i_dec2)

        # decoder level 1
        i_dec1 = self.ID_block1(i_dec2, i_jump0)

        # IMPORTANT:
        # hv_1 here is the internal enhanced HV feature after the enhancement network.
        # This is what we use for strict CDA alignment.
        hv_1 = self.HVD_block1(hv_2, hv_jump0)
        pred_hv_feat_cda = self.cda_pred_head(hv_1)

        # final image-space H / V / I
        i_dec0 = self.ID_block0(i_dec1)
        hv_0 = self.HVD_block0(hv_1)

        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        if return_aux:
            aux = {
                "output_hvi": output_hvi,                 # final internal HVI
                "pred_hv_feat_cda": pred_hv_feat_cda,     # internal enhanced HV feature in CDA space
                "l_vcf": l_vcf,
            }
            return output_rgb, aux

        return output_rgb

    def HVIT(self, x):
        return self.trans.HVIT(x)

    def get_cda_gt_feature(self, gt_rgb):
        """
        Map GT HV to the same CDA color feature space.
        """
        gt_hvi = self.trans.HVIT(gt_rgb)
        gt_hv = gt_hvi[:, :2, :, :]
        gt_hv_feat = self.cda_gt_head(gt_hv)
        return gt_hv_feat