from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = [
    "PaddingResidual",
    "C2",
    "C2f",
    "C3",
    "C3x",
    "C3Ghost",
    "BottleneckCSP",
    "RepBottleneck",
    "RepC3",
    "C3k2M",
    "C3k",
]

# -------------------------- 1. 新增：替换原始Bottleneck的四方向填充残差模块 --------------------------
class PaddingResidual(nn.Module):
    """
    四方向填充残差模块（替换原始Bottleneck）：
    1. 四方向（左上/右上/左下/右下）填充2行/列0
    2. 3x3卷积保持尺寸不变，拼接后1x1降维
    3. 与原始输入残差连接，兼容原Bottleneck参数接口
    """

    def __init__(self, in_channels: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3),
                 e: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.shortcut = shortcut
        self.add = shortcut and True  # 输入输出通道一致，支持残差连接

        # 降维比例：e=0.5时，中间通道数为in_channels的一半
        self.mid_channels = int(in_channels * e)  # 中间通道数（压缩后）

        # 四方向定向填充（左/右/上/下）
        self.pad_lt = nn.ConstantPad2d((2, 0, 2, 0), 0)  # 左上：左2+上2
        self.pad_rt = nn.ConstantPad2d((0, 2, 2, 0), 0)  # 右上：右2+上2
        self.pad_lb = nn.ConstantPad2d((2, 0, 0, 2), 0)  # 左下：左2+下2
        self.pad_rb = nn.ConstantPad2d((0, 2, 0, 2), 0)  # 右下：右2+下2

        # 每个分支：1×1降维 → 3×3卷积（分组卷积兼容）
        # 1×1卷积将通道从in_channels→mid_channels（降维）
        # 3×3卷积保持mid_channels通道（特征提取）
        self.conv1x1_lt = Conv(in_channels, self.mid_channels, k=1, g=g)
        self.conv3x3_lt = Conv(self.mid_channels, self.mid_channels, k=3, p=0, g=g)

        self.conv1x1_rt = Conv(in_channels, self.mid_channels, k=1, g=g)
        self.conv3x3_rt = Conv(self.mid_channels, self.mid_channels, k=3, p=0, g=g)

        self.conv1x1_lb = Conv(in_channels, self.mid_channels, k=1, g=g)
        self.conv3x3_lb = Conv(self.mid_channels, self.mid_channels, k=3, p=0, g=g)

        self.conv1x1_rb = Conv(in_channels, self.mid_channels, k=1, g=g)
        self.conv3x3_rb = Conv(self.mid_channels, self.mid_channels, k=3, p=0, g=g)

        # 1×1升维：4个分支拼接（4×mid_channels）→ 原始通道数（in_channels）
        self.conv1x1_up = Conv(4 * self.mid_channels, in_channels, k=1, g=g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x  # 残差连接的原始输入

        # 四个分支处理：填充→1×1降维→3×3卷积
        # 左上分支
        x_lt = self.pad_lt(x)
        x_lt = self.conv1x1_lt(x_lt)  # 降维到mid_channels
        y_lt = self.conv3x3_lt(x_lt)  # 3×3特征提取

        # 右上分支
        x_rt = self.pad_rt(x)
        x_rt = self.conv1x1_rt(x_rt)
        y_rt = self.conv3x3_rt(x_rt)

        # 左下分支
        x_lb = self.pad_lb(x)
        x_lb = self.conv1x1_lb(x_lb)
        y_lb = self.conv3x3_lb(x_lb)

        # 右下分支
        x_rb = self.pad_rb(x)
        x_rb = self.conv1x1_rb(x_rb)
        y_rb = self.conv3x3_rb(x_rb)

        # 拼接所有分支（通道维度：4×mid_channels）
        y_cat = torch.cat([y_lt, y_rt, y_lb, y_rb], dim=1)

        # 1×1升维回原始通道数
        y = self.conv1x1_up(y_cat)

        # 残差连接
        return y + residual if self.add else y


# -------------------------- 2. 修改：依赖Bottleneck的C2模块 --------------------------
class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions（内部Bottleneck→PaddingResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        # 替换：Bottleneck序列 → PaddingResidual序列
        self.m = nn.Sequential(*(PaddingResidual(in_channels=self.c, shortcut=shortcut, g=g) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


# -------------------------- 3. 修改：依赖Bottleneck的C2f模块 --------------------------
class C2f(nn.Module):
    """Faster CSP Bottleneck with 2 convolutions（内部Bottleneck→PaddingResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # 替换：Bottleneck列表 → PaddingResidual列表
        self.m = nn.ModuleList(PaddingResidual(in_channels=self.c, shortcut=shortcut, g=g) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# -------------------------- 4. 修改：依赖Bottleneck的C3模块 --------------------------
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions（内部Bottleneck→PaddingResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        # 替换：Bottleneck序列 → PaddingResidual序列
        self.m = nn.Sequential(*(PaddingResidual(in_channels=c_, shortcut=shortcut, g=g) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# -------------------------- 5. 修改：依赖Bottleneck的C3x模块 --------------------------
class C3x(C3):
    """C3 module with cross-convolutions（内部Bottleneck→PaddingResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        # 替换：Bottleneck序列 → PaddingResidual序列（保留cross-conv的k参数兼容）
        self.m = nn.Sequential(*(PaddingResidual(in_channels=self.c_, shortcut=shortcut, g=g) for _ in range(n)))


# -------------------------- 6. 修改：依赖Bottleneck的C3Ghost模块 --------------------------
class C3Ghost(C3):
    """C3 module with GhostBottleneck（内部Bottleneck→PaddingResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        # 替换：GhostBottleneck序列 → PaddingResidual序列
        self.m = nn.Sequential(*(PaddingResidual(in_channels=self.c_, shortcut=shortcut, g=g) for _ in range(n)))


# -------------------------- 7. 修改：依赖Bottleneck的BottleneckCSP模块 --------------------------
class BottleneckCSP(nn.Module):
    """CSP Bottleneck（内部Bottleneck→PaddingResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        # 替换：Bottleneck序列 → PaddingResidual序列
        self.m = nn.Sequential(*(PaddingResidual(in_channels=c_, shortcut=shortcut, g=g) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


# -------------------------- 8. 修改：依赖Bottleneck的RepBottleneck模块 --------------------------
class RepBottleneck(PaddingResidual):
    """Rep bottleneck（继承自PaddingResidual，替换原Bottleneck父类）"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3),
                 e: float = 0.5):
        # 直接继承PaddingResidual，保留RepBottleneck原有接口
        super().__init__(in_channels=c1, shortcut=shortcut, g=g, k=k, e=e)
        # 兼容原RepBottleneck的cv1（RepConv），替换PaddingResidual的conv3x3_lt为RepConv
        self.conv3x3_lt = RepConv(c1, c1, k[0], 1)


# -------------------------- 9. 修改：依赖Bottleneck的RepC3模块 --------------------------
class RepC3(nn.Module):
    """Rep C3（内部Bottleneck→PaddingResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # 替换：RepConv序列 → PaddingResidual序列
        self.m = nn.Sequential(*[PaddingResidual(in_channels=c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


# -------------------------- 10. 修改：依赖Bottleneck的C3k2模块 --------------------------
class DireC3(C2f):
    """Faster CSP Bottleneck with 2 convolutions（内部Bottleneck→PaddingResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1,
                 shortcut: bool = True):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 替换：C3k/Bottleneck → PaddingResidual
        self.m = nn.ModuleList(PaddingResidual(in_channels=self.c, shortcut=shortcut, g=g) for _ in range(n))


# -------------------------- 11. 修改：依赖Bottleneck的C3k模块 --------------------------
class C3k(C3):
    """C3k module with customizable kernel（内部Bottleneck→PaddingResidual）"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        # 替换：Bottleneck序列 → PaddingResidual序列（保留k参数兼容）
        self.m = nn.Sequential(*(PaddingResidual(in_channels=self.c_, shortcut=shortcut, g=g) for _ in range(n)))

