import torch
import torch.nn as nn
import torch.nn.functional as F

class Dysample(nn.Module):
    
    def __init__(self, in_channels, out_channels, use_asymmetric=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_asymmetric = use_asymmetric  # 是否使用非对称卷积（适合船舶）
        
        # 1. 轻量级通道适配（必要时）
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.proj = nn.Identity()
        
        # 2. 智能多尺度特征提取（核心改进）
        # 针对船舶特性：长条形状 + 多尺度
        self.multi_scale = self._build_multi_scale(out_channels, use_asymmetric)
        
        # 3. 注意力增强模块（轻量但高效）
        self.attention = self._build_attention(out_channels)
        
        # 4. 高效下采样（保留关键信息）
        self.downsample = self._build_downsample(out_channels)
        
        # 5. 智能残差连接
        self.residual = self._build_residual(in_channels, out_channels)
        
        # 6. 特征融合（可选）
        self.fusion = nn.Conv2d(out_channels, out_channels, 1, bias=False) if use_asymmetric else nn.Identity()
        
        self.final_activation = nn.SiLU(inplace=True)
    
    def _build_multi_scale(self, channels, asymmetric):
        """构建多尺度特征提取模块"""
        modules = nn.ModuleList()
        
        if asymmetric:
            # 非对称卷积核，适合船舶长条特征
            # 分支1：3×1 + 1×3 替代 3×3（减少计算，增强长条特征）
            modules.append(nn.Sequential(
                nn.Conv2d(channels, channels//4, (3, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(channels//4),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels//4, channels//4, (1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(channels//4),
                nn.SiLU(inplace=True)
            ))
            
            # 分支2：5×1 + 1×5（更大的感受野）
            modules.append(nn.Sequential(
                nn.Conv2d(channels, channels//4, (5, 1), padding=(2, 0), bias=False),
                nn.BatchNorm2d(channels//4),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels//4, channels//4, (1, 5), padding=(0, 2), bias=False),
                nn.BatchNorm2d(channels//4),
                nn.SiLU(inplace=True)
            ))
        else:
            # 对称卷积核（通用场景）
            modules.append(nn.Sequential(
                nn.Conv2d(channels, channels//2, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels//2),
                nn.SiLU(inplace=True)
            ))
        
        # 空洞卷积（扩大感受野）
        modules.append(nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels//2),
            nn.SiLU(inplace=True)
        ))
        
        return modules
    
    def _build_attention(self, channels):
        """构建轻量但有效的注意力模块"""
        return nn.Sequential(
            # 通道注意力（轻量版SE）
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels//16), 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(4, channels//16), channels, 1, bias=False),
            nn.Sigmoid(),
            
            # 空间注意力（轻量）
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
    
    def _build_downsample(self, channels):
        """构建高效下采样模块"""
        # 使用深度可分离卷积 + 点卷积
        return nn.Sequential(
            # 深度卷积（空间下采样）
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            
            # 点卷积（通道交互）
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
    
    def _build_residual(self, in_channels, out_channels):
        """构建智能残差连接"""
        if in_channels == out_channels:
            return nn.Sequential(
                nn.AvgPool2d(2),
                nn.Identity()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.AvgPool2d(2)
            )
    
    def forward(self, x):
        identity = x
        
        # 1. 通道适配
        x = self.proj(x)
        
        # 2. 多尺度特征提取（并行）
        multi_features = []
        for module in self.multi_scale:
            multi_features.append(module(x))
        
        # 融合多尺度特征
        if len(multi_features) > 1:
            x_multi = torch.cat(multi_features, dim=1)
            if isinstance(self.fusion, nn.Conv2d):
                x_multi = self.fusion(x_multi)
        else:
            x_multi = multi_features[0]
        
        # 3. 注意力增强
        att = self.attention(x_multi)
        x_att = x_multi * att
        
        # 4. 下采样
        x_down = self.downsample(x_att)
        
        # 5. 残差连接
        x_res = self.residual(identity)
        
        # 6. 最终输出
        out = x_down + x_res
        return self.final_activation(out)
