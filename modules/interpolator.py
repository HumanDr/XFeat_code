"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions.
    高效的稀疏2D位置插值模块
    功能：在给定的稀疏2D坐标位置上对输入特征图进行插值采样
    """
    def __init__(self, mode = 'bicubic', align_corners = False):
        """
        初始化方法
        参数：
            mode: 插值模式，默认为'bicubic'（双三次插值）
            align_corners: 是否对齐角点，默认为False
        """
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """
        前向传播方法
        参数：
            x: 输入特征图 [B, C, H, W]
            pos: 要采样的2D坐标 [B, N, 2]
            H, W: 原始分辨率，用于坐标归一化
        返回：
            采样后的特征 [B, N, C]
        """
        """ Normalize coords to [-1,1]. """
        # 归一化公式：2*(x/(size-1)) - 1
        # 创建归一化因子张量[W-1, H-1]，保持与输入x相同的设备和数据类型
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        """
        前向传播方法
        参数：
            x: 输入特征图 [B, C, H, W]
            pos: 要采样的2D坐标 [B, N, 2]
            H, W: 原始分辨率，用于坐标归一化
        返回：
            采样后的特征 [B, N, C]
        """
        # 1. 坐标归一化并调整维度
        # 将坐标归一化到[-1,1]范围，并增加一个维度 [B, N, 1, 2] 以适应grid_sample要求
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        # 2. 使用grid_sample进行插值采样
        # 输入x: [B, C, H, W]
        # 输出: [B, C, N, 1] (因为grid是[B, N, 1, 2])
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        # 3. 调整输出维度并压缩不必要的维度
        # permute将维度从[B, C, N, 1]变为[B, N, 1, C]
        # squeeze(-2)移除倒数第二个维度，得到[B, N, C]
        return x.permute(0,2,3,1).squeeze(-2)

feature_map = torch.randn(1, 256, 64, 64)  # [B, C, H, W]
print(feature_map.shape)
keypoints = torch.tensor([[[10, 20], [30, 40],[10,12]]])  # [B, N, 2]
print(keypoints.shape)
interpolator = InterpolateSparse2d(mode='bicubic')
print(interpolator(feature_map, keypoints, H=64, W=64).shape)
sampled_features = interpolator(feature_map, keypoints, H=64, W=64)
print(sampled_features)
print(sampled_features.shape)