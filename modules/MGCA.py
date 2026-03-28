
class MGCA(nn.Module):
    """
    MCA-SO: Moment Channel Attention for Small-Object detection (lightweight).
    - channels_A: channels of guiding feature A (used to compute moments)
    - channels_B: channels of target feature B (to be weighted)
    - reduction: internal bottleneck for channel_map
    Design:
      1) compute per-channel mean, variance, high-freq energy from A
      2) per-channel linear combinaion of these 3 moments (learned, tiny params)
      3) optional normalization (batch-wise) to stabilize across scales
      4) small 1x1 channel_map: CA -> mid -> CB
      5) sigmoid -> apply on B: out = B + B * att
    """
    def __init__(self, channels_A: int, channels_B: int, reduction: int = 8, eps: float = 1e-6):
        super().__init__()
        self.CA = channels_A
        self.CB = channels_B
        self.eps = eps

        # ------ per-channel linear combiner for 3 moments (mean,var,hf) ------
        # weight: (CA, 3)  bias: (CA,)
        # very small number of params: 3 * CA
        self.moments_w = nn.Parameter(torch.randn(channels_A, 3) * 0.1)
        self.moments_b = nn.Parameter(torch.zeros(channels_A))

        # ------ channel mapping (lightweight) CA -> mid -> CB ------
        mid = max(8, channels_A // reduction)
        self.channel_map = nn.Sequential(
            nn.Conv2d(channels_A, mid, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels_B, kernel_size=1, bias=True)
        )

        # small layernorm on moments descriptor (stabilize across scales)
        self.moments_norm = nn.LayerNorm([channels_A])  # normalize per-channel vector

        self.sigmoid = nn.Sigmoid()

        # init small
        nn.init.normal_(self.moments_w, mean=0.0, std=0.05)
        nn.init.zeros_(self.moments_b)
        for m in self.channel_map.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, A, B):
        """
        A: (N, CA, H, W)  guiding feature (shallow or deep depending)
        B: (N, CB, H, W)  target feature (same spatial size as A expected; if not, A will be pooled)
        Returns:
          out: (N, CB, H, W)
        Notes:
          - If A and B have different spatial sizes, we adapt: compute moments on A (native),
            produce per-channel descriptors, then apply to B (B's spatial dims unaffected).
        """
        N, CA, HA, WA = A.shape
        N2, CB, HB, WB = B.shape

        # If spatial sizes differ, adapt: compute moments on A's native spatial size.
        # Moments: mean, variance, high-frequency energy
        # 1) mean
        mean = A.mean(dim=[2, 3])                 # (N, CA)
        # 2) variance
        var = ((A - mean.view(N, CA, 1, 1)) ** 2).mean(dim=[2, 3])  # (N, CA)
        # 3) high-frequency energy: HF = mean((A - local_mean)^2)
        # compute local mean via 3x3 avg (same padding) then high freq = (A - local_mean)^2 mean
        local_mean = F.avg_pool2d(A, kernel_size=3, stride=1, padding=1)
        hf = ((A - local_mean) ** 2).mean(dim=[2, 3])  # (N, CA)

        # stack moments per-channel: (N, CA, 3)
        moments = torch.stack([mean, var, hf], dim=2)  # (N, CA, 3)

        # per-channel linear combination using tiny params (CA x 3)
        # weights: (CA, 3) -> compute dot over last dim
        # we compute: desc[n, c] = sum_k moments[n,c,k] * w[c,k] + b[c]
        # using einsum
        desc = torch.einsum('nck,ck->nc', moments, self.moments_w) + self.moments_b  # (N, CA)

        # normalize descriptor per-sample to stabilize across scales and batches
        # LayerNorm over channel dimension: expects shape (N, CA). We expand dims accordingly
        desc = self.moments_norm(desc)  # (N, CA)

        # reshape to (N, CA, 1, 1) to feed channel_map
        desc = desc.view(N, CA, 1, 1)

        # channel_map: CA -> mid -> CB
        att_logits = self.channel_map(desc)   # (N, CB, 1, 1)
        att = self.sigmoid(att_logits)        # (N, CB, 1, 1)

        # apply attention to B; broadcast spatially
        out = B + B * att

        return out


