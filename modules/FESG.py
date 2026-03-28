class SimAM(nn.Module):
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        var  = ((x - mean) ** 2).mean(dim=[2, 3], keepdim=True)
        e = (x - mean) ** 2 / (4 * (var + self.e_lambda)) + 0.5
        att = torch.sigmoid(e.mean(dim=1, keepdim=True))
        return att


class EASA_Module(nn.Module):
    """
    Lightweight EASA-style spatial self-modulation for A -> attention map.
    Produces 1-channel spatial attention from A.
    """
    def __init__(self, channels_A, mid=None):
        super().__init__()
        if mid is None:
            mid = max(16, channels_A // 8)
        self.dw = nn.Conv2d(channels_A, channels_A, kernel_size=3, padding=1,
                            groups=channels_A, bias=False)
        self.pw1 = nn.Conv2d(channels_A, mid, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.pw2 = nn.Conv2d(mid, 1, kernel_size=1, bias=True)

        # init
        for m in (self.dw, self.pw1, self.pw2):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, A):
        x = self.dw(A)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)       # (N,1,H,W)
        att = torch.sigmoid(x)
        return att


class FrequencySeparator(nn.Module):
    """
    Frequency separation: lowpass (fixed small blur) + high = x - low.
    - lowpass implemented as depthwise average-like conv (fixed weights)
    """
    def __init__(self, channels, kernel_size: int = 3):
        super().__init__()
        assert kernel_size in (3, 5)
        pad = kernel_size // 2
        # create depthwise conv with fixed averaging weights
        self.low_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                                  padding=pad, groups=channels, bias=False)
        # build average kernel for each channel
        k = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32) / (kernel_size * kernel_size)
        with torch.no_grad():
            self.low_conv.weight.copy_(k.repeat(channels, 1, 1, 1))
        # freeze weights (optional: keep false if want learnable lowpass)
        self.low_conv.weight.requires_grad = False

    def forward(self, x):
        low = self.low_conv(x)      # low-frequency component (smoothed)
        high = x - low              # high-frequency component (detail)
        return low, high


class EdgeEnhanceDW(nn.Module):
    """
    Depthwise Laplacian edge enhancer with learnable scale (initialized small).
    """
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                            groups=channels, bias=True)
        lap = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], dtype=torch.float32)
        lap = lap.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            self.dw.weight.copy_(lap.repeat(channels, 1, 1, 1))
            self.dw.bias.zero_()
        # learnable multiplier for stability (small init)
        self.scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, x):
        return self.scale * self.dw(x)


class LowEnhancePW(nn.Module):
    """
    Low-frequency enhancement: small pointwise projection to re-introduce structural cues.
    """
    def __init__(self, channels_low, mid=32):
        super().__init__()
        self.pw = nn.Conv2d(channels_low, channels_low, kernel_size=1, bias=True)
        # small residual-like init
        nn.init.zeros_(self.pw.weight)
        if self.pw.bias is not None:
            nn.init.zeros_(self.pw.bias)
        # a small scale
        self.scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, low):
        return self.scale * self.pw(low)


class FESG(nn.Module):
    """
    EASA spatial attention + Frequency separation + Edge enhancement.
    Input:
      A: guiding feature (N, CA, H1, W1)
      B: target feature  (N, CB, H2, W2)
    Output:
      out: same shape as B
    Formula:
      att = EASA(A)  -> resized to B
      BW = B * att
      low, high = freq_sep(BW)
      low_enh = LowEnhance(low)
      high_enh = EdgeEnhance(high)
      out = B + BW + low_enh + high_enh
    Notes:
      - lowpass kernel fixed (avg) by default; you can set learnable if desired.
      - edge scale and low scale are learnable; initialized small for stability.
    """
    def __init__(self, channels_A, channels_B, low_kernel=3, mid=None):
        super().__init__()
        self.easa = EASA_Module(channels_A, mid=mid)
        self.simam = SimAM()
        self.to_att = nn.Conv2d(1, 1, 1, bias=True)  # tiny combining projection (could be identity)
        # freq sep on B
        self.freq = FrequencySeparator(channels_B, kernel_size=low_kernel)
        self.low_enh = LowEnhancePW(channels_B, mid=32)
        self.high_enh = EdgeEnhanceDW(channels_B)

        # combine projector (optional) for BW -> keep small param overhead
        self.combine_pw = nn.Conv2d(channels_B, channels_B, kernel_size=1, bias=False)
        nn.init.zeros_(self.combine_pw.weight)  # start as identity-ish but small

    def forward(self, A, B):
        # 1) attention from A (EASA), then optional SimAM refine and combine
        att_a = self.easa(A)                       # (N,1,H1,W1)
        # refine att via SimAM applied on A's modulated features if desired:
        # we can also use SimAM on A to modulate att -- here we combine
        sim = self.simam(A)                        # (N,1,H1,W1)
        att = torch.sigmoid(att_a * sim)          # refine
        att = F.interpolate(att, size=B.shape[2:], mode='bilinear', align_corners=False)

        # 2) apply attention to B
        BW = B * att                               # (N,CB,H2,W2)

        # 3) optional lightweight combine PW (keeps BW compatible)
        BW = self.combine_pw(BW) + BW              # small projection + residual

        # 4) frequency separation on BW
        low, high = self.freq(BW)                  # both shape (N,CB,H2,W2)

        # 5) low / high enhancement
        low_enh = self.low_enh(low)                # (N,CB,H2,W2)
        high_enh = self.high_enh(high)             # (N,CB,H2,W2)

        # 6) final aggregation: keep residual B + attentioned BW + both enhancers
        out = B + BW + low_enh + high_enh

        return out

