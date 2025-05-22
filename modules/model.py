"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	  基础卷积块：Conv2d -> BatchNorm -> ReLU
    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size/padding/stride/dilation: 卷积参数
        bias: 是否使用偏置
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False), # 无学习参数的BN
									  nn.ReLU(inplace = True), # 原地操作，节省内存
									)

	def forward(self, x):
	  return self.layer(x)

class XFeatModel(nn.Module):
	"""
	   Implementation of architecture described in 
	   "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	"""

	def __init__(self):
		super().__init__()
		self.norm = nn.InstanceNorm2d(1) # 输入标准化（灰度图单通道）


		########### ⬇️ CNN Backbone & Heads ⬇️ ###########
		# ----- 主干网络（多尺度特征提取）-----
		self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4), # 下采样4倍
			  						 nn.Conv2d (1, 24, 1, stride = 1, padding=0) ) # 1x1卷积降维

		# Block1-5: 多级特征提取（逐步下采样）
		self.block1 = nn.Sequential(
										BasicLayer( 1,  4, stride=1), # 保持分辨率
										BasicLayer( 4,  8, stride=2), # 下采样2倍
										BasicLayer( 8,  8, stride=1), # 保持分辨率
										BasicLayer( 8, 24, stride=2), # 累计4倍下采样
									) # 累计下采样：1 × 2 × 1 × 2 = 4倍 看stride

		self.block2 = nn.Sequential(
										BasicLayer(24, 24, stride=1), # 保持分辨率
										BasicLayer(24, 24, stride=1),
									 )  # 保持4倍下采样

		self.block3 = nn.Sequential(
										BasicLayer(24, 64, stride=2),
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, 1, padding=0),
									 ) 	# 累计：4 × 2 = 8倍
		self.block4 = nn.Sequential(
										BasicLayer(64, 64, stride=2),
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, stride=1),
									 )  # 累计：8 × 2 = 16倍

		self.block5 = nn.Sequential(
										BasicLayer( 64, 128, stride=2),
										BasicLayer(128, 128, stride=1),
										BasicLayer(128, 128, stride=1),
										BasicLayer(128,  64, 1, padding=0),
									 )  # 累计：16 × 2 = 32倍

		# ----- 特征融合与输出头 -----
		self.block_fusion =  nn.Sequential(
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, stride=1),
										nn.Conv2d (64, 64, 1, padding=0) # 1x1卷积压缩特征
									 )

		self.heatmap_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 1, 1),
										nn.Sigmoid()
									)

		# 热图头（可靠性预测）
		self.keypoint_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 65, 1),
									)


  		########### ⬇️ Fine Matcher MLP ⬇️ ###########
		# ----- 精细匹配MLP（描述子增强）-----
		# ...（4层MLP，逐步降维到64）
		self.fine_matcher =  nn.Sequential(
											nn.Linear(128, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 64),
										)

	def _unfold2d(self, x, ws = 2):
		"""
			Unfolds tensor in 2D with desired ws (window size) and concat the channels
			将输入张量按窗口展开并拼接通道
			参数：
				x: 输入张量 (B,C,H,W)
				ws: 窗口大小
			返回：
				(B, C*ws², H/ws, W/ws)
		"""
		B, C, H, W = x.shape
		x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
			.reshape(B, C, H//ws, W//ws, ws**2)
		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


	def forward(self, x):
		"""
			input:
				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
			return:
				feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
				keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
				heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

		"""
		#dont backprop through normalization
		# 1. 输入预处理（无梯度）
		with torch.no_grad():
			x = x.mean(dim=1, keepdim = True)  # RGB转灰度
			x = self.norm(x)	 # 实例归一化

		#main backbone
		x1 = self.block1(x)  # 4倍下采样
		x2 = self.block2(x1 + self.skip1(x))  # 残差连接+4倍下采样
		x3 = self.block3(x2)	# 8倍下采样
		x4 = self.block4(x3)	# 16倍下采样
		x5 = self.block5(x4)	# 32倍下采样

		#pyramid fusion
		# 3. 特征金字塔融合
		# x3.shape[-2]和x3.shape[-1]分别表示x3的高度（H）和宽度（W）
		# 采用双线性插值算法，通过对相邻4个像素的加权平均计算新像素值，适合特征图的上采样。
		# interpolate是张量上采样/下采样的函数，在这里的作用是将深层特征图（x4和x5）的分辨率调整到与中层特征图（x3）相同的尺寸
		x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear') # 上采样对齐分辨率
		x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear') # 上采样对齐分辨率
		feats = self.block_fusion( x3 + x4 + x5 )  # 多尺度特征相加

		#heads
		heatmap = self.heatmap_head(feats) # Reliability map 可靠性热图
		keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits    关键点热图

		return feats, keypoints, heatmap # 特征图/关键点logits/热图

