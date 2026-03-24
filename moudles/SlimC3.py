from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = [
    "MultiKernelResidual",  # 新增：多尺度非方形卷积残差模块
    "C2",
    "C2f",
    "C3",
    "C3x",
    "C3Ghost",
    "BottleneckCSP",
    "RepBottleneck",
    "RepC3",
    "C3k2S",
    "C3k",
]


# -------------------------- 1. 核心修改：多尺度非方形卷积残差模块 --------------------------
class MultiKernelResidual(nn.Module):
    """
    3×5 + 3×7非方形卷积并联残差模块：
    1. 双分支非方形卷积（适配细长物体长轴特征）
    2. 1×1卷积先降维再升维，控制参数量
    3. 与原始输入残差连接，兼容原Bottleneck接口
    """

    def __init__(self, in_channels: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.shortcut = shortcut
        self.add = shortcut and True  # 输入输出通道一致，支持残差

        # 通道缩放比例（e=0.5时，中间通道为in_channels//2）
        mid_channels = int(in_channels * e)

        # 分支1：1×1降维 → 3×5卷积（padding=(1,2)保持尺寸，适配长轴）
        self.branch1 = nn.Sequential(
            Conv(in_channels, mid_channels, k=1, g=g),  # 降维
            Conv(mid_channels, mid_channels, k=(3, 5), p=(1, 2), g=g)  # 3×5卷积，H方向pad1，W方向pad2
        )

        # 分支2：1×1降维 → 3×7卷积（padding=(1,3)保持尺寸，适配更长轴）
        self.branch2 = nn.Sequential(
            Conv(in_channels, mid_channels, k=1, g=g),  # 降维
            Conv(mid_channels, mid_channels, k=(3, 7), p=(1, 3), g=g)  # 3×7卷积，H方向pad1，W方向pad3
        )

        # 1×1卷积升维（双分支拼接后→原通道数）
        self.conv1x1 = Conv(2 * mid_channels, in_channels, k=1, g=g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # 双分支特征提取
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        # 拼接+升维+残差连接
        y_cat = torch.cat([y1, y2], dim=1)
        y = self.conv1x1(y_cat)
        return y + residual if self.add else y


# -------------------------- 2. 适配修改：依赖Bottleneck的各模块 --------------------------
class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions（内部替换为MultiKernelResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        # 替换为MultiKernelResidual序列
        self.m = nn.Sequential(
            *(MultiKernelResidual(in_channels=self.c, shortcut=shortcut, g=g, e=e) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster CSP Bottleneck with 2 convolutions（内部替换为MultiKernelResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # 替换为MultiKernelResidual列表
        self.m = nn.ModuleList(MultiKernelResidual(in_channels=self.c, shortcut=shortcut, g=g, e=e) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions（内部替换为MultiKernelResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        # 替换为MultiKernelResidual序列
        self.m = nn.Sequential(*(MultiKernelResidual(in_channels=c_, shortcut=shortcut, g=g, e=e) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions（内部替换为MultiKernelResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        # 替换为MultiKernelResidual序列
        self.m = nn.Sequential(
            *(MultiKernelResidual(in_channels=self.c_, shortcut=shortcut, g=g, e=e) for _ in range(n)))


class C3Ghost(C3):
    """C3 module with GhostBottleneck（内部替换为MultiKernelResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        # 替换为MultiKernelResidual序列
        self.m = nn.Sequential(
            *(MultiKernelResidual(in_channels=self.c_, shortcut=shortcut, g=g, e=e) for _ in range(n)))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck（内部替换为MultiKernelResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        # 替换为MultiKernelResidual序列
        self.m = nn.Sequential(*(MultiKernelResidual(in_channels=c_, shortcut=shortcut, g=g, e=e) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class RepBottleneck(MultiKernelResidual):
    """Rep bottleneck（继承自MultiKernelResidual）"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3),
                 e: float = 0.5):
        # 继承MultiKernelResidual，兼容RepBottleneck接口
        super().__init__(in_channels=c1, shortcut=shortcut, g=g, e=e)
        # 替换分支1的3×5卷积为RepConv（重参数化卷积，提升推理速度）
        mid_channels = int(c1 * e)
        self.branch1[1] = RepConv(mid_channels, mid_channels, k=(3, 5), p=(1, 2), g=g)


class RepC3(nn.Module):
    """Rep C3（内部替换为MultiKernelResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # 替换为MultiKernelResidual序列
        self.m = nn.Sequential(*[MultiKernelResidual(in_channels=c_, e=e) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class SlimC3(C2f):
    """Faster CSP Bottleneck with 2 convolutions（内部替换为MultiKernelResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1,
                 shortcut: bool = True):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 替换为MultiKernelResidual列表
        self.m = nn.ModuleList(MultiKernelResidual(in_channels=self.c, shortcut=shortcut, g=g, e=e) for _ in range(n))


class C3k(C3):
    """C3k module with customizable kernel（内部替换为MultiKernelResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        # 替换为MultiKernelResidual序列
        self.m = nn.Sequential(
            *(MultiKernelResidual(in_channels=self.c_, shortcut=shortcut, g=g, e=e) for _ in range(n)))
