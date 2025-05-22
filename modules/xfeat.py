
"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import numpy as np
import os
import torch
import torch.nn.functional as F

import tqdm

from modules.model import *
from modules.interpolator import InterpolateSparse2d

class XFeat(nn.Module):
	""" 
		Implements the inference module for XFeat. 
		It supports inference for both sparse and semi-dense feature extraction & matching.
	"""

	def __init__(self, weights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat.pt', top_k = 4096, detection_threshold=0.05):
		"""
			weights (str or dict, 可选): 模型权重文件的路径或预加载的权重字典，默认为项目 weights 目录下的 xfeat.pt 文件。
        	top_k (int, 可选): 保留的最佳特征数量，默认为 4096。
			detection_threshold (float, 可选): 特征检测的阈值，默认为 0.05。
		"""
		super().__init__()
		self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.net = XFeatModel().to(self.dev).eval()
		self.top_k = top_k
		self.detection_threshold = detection_threshold

		if weights is not None: # 如果权重不为空
			if isinstance(weights, str): # 如果权重是文件路径
				print('loading weights from: ' + weights)
				# 从指定路径加载模型权重并映射到指定设备
				self.net.load_state_dict(torch.load(weights, map_location=self.dev))
			else: # 如果权重是字典
				self.net.load_state_dict(weights) # 直接加载权重字典

		# 初始化插值器，使用双三次插值方法
		self.interpolator = InterpolateSparse2d('bicubic')

		#Try to import LightGlue from Kornia
		self.kornia_available = False
		self.lighterglue = None # 初始化 LightGlue 模型为 None，后续根据需要加载
		try:
			import kornia # 尝试导入 kornia 库
			self.kornia_available=True # 若导入成功，将 kornia 可用标志设置为 True
		except:
			pass


	@torch.inference_mode() # 开启推理模式，禁用梯度计算以提高推理速度 禁用梯度可减少内存占用，提升速度（尤其是对实时应用）。
	def detectAndCompute(self, x, top_k = None, detection_threshold = None):
		"""
			Compute sparse keypoints & descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
				x (torch.Tensor(B, C, H, W)): 灰度图或 RGB 图像
                    top_k (int, 可选): 保留最佳的 k 个特征，默认为类初始化时的 top_k 值
                    detection_threshold (float, 可选): 特征检测阈值，默认为类初始化时的 detection_threshold 值
			return:
				List[Dict]:
					'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
					'scores'       ->   torch.Tensor(N,): keypoint scores
					'descriptors'  ->   torch.Tensor(N, 64): local features
				 List[Dict]:
                        'keypoints'    ->   torch.Tensor(N, 2): 关键点坐标 (x, y)
                        'scores'       ->   torch.Tensor(N,): 关键点分数
                        'descriptors'  ->   torch.Tensor(N, 64): 局部特征描述子
		"""
		"""
		计算稀疏关键点及其描述符（支持批量处理）。

		输入:
			x -> torch.Tensor(B, C, H, W): 灰度或RGB图像（Batch, 通道, 高度, 宽度）
			top_k -> int: 保留的最佳特征点数量（若为None则使用类默认值self.top_k）
			multiscale -> bool: 是否启用多尺度特征提取（默认为True）

		返回: 按置信度排序的特征（从高到低）
			List[Dict]: 每个样本的字典包含:
				'keypoints'   -> torch.Tensor(top_k, 2): 粗粒度关键点坐标(x,y)
				'scales'      -> torch.Tensor(top_k,): 特征点提取时的尺度（仅多尺度模式下有效）
				'descriptors' -> torch.Tensor(top_k, 64): 粗粒度局部特征描述符
		"""
		if top_k is None: top_k = self.top_k # 若未指定 top_k，使用类初始化时的 top_k 值
		if detection_threshold is None: detection_threshold = self.detection_threshold # 若未指定检测阈值，使用类初始化时的阈值
		x, rh1, rw1 = self.preprocess_tensor(x)  # 对输入图像进行预处理，确保图像尺寸能被 32 整除

		B, _, _H1, _W1 = x.shape # 获取输入图像的批量大小、通道数、高度和宽度
        
		M1, K1, H1 = self.net(x) # 将预处理后的图像输入模型，得到特征图 M1、关键点图 K1 和可靠性图 H1
		M1 = F.normalize(M1, dim=1) # 对特征图 M1 进行 L2 归一化处理

		#Convert logits to heatmap and extract kpts
		K1h = self.get_kpts_heatmap(K1) # 将关键点图转换为热图
		mkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5) # 非极大值抑制获取关键点

		#Compute reliability scores
		_nearest = InterpolateSparse2d('nearest') # 最近邻插值器
		_bilinear = InterpolateSparse2d('bilinear') # 双线性插值器
		scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)  # 计算关键点的可靠性分数
		scores[torch.all(mkpts == 0, dim=-1)] = -1 # 无效关键点设为-1

		#Select top-k features
		# 按分数排序并选择top-k特征
		idxs = torch.argsort(-scores)  # 降序排列索引
		mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)[:, :top_k] 	# 选取x坐标
		mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)[:, :top_k]	# 选取y坐标
		mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1) # 合并x和y坐标
		scores = torch.gather(scores, -1, idxs)[:, :top_k] # 选取对应的分数

		#Interpolate descriptors at kpts positions
		feats = self.interpolator(M1, mkpts, H = _H1, W = _W1) 	# 双线性插值获取特征描述子

		#L2-Normalize
		feats = F.normalize(feats, dim=-1) 	  # 对特征描述子进行 L2 归一化处理

		#Correct kpt scale
		mkpts = mkpts * torch.tensor([rw1,rh1], device=mkpts.device).view(1, 1, -1)  # 校正关键点的尺度

		valid = scores > 0  # 筛选出分数大于0的有效关键点
		return [  
				   {'keypoints': mkpts[b][valid[b]],
					'scores': scores[b][valid[b]],
					'descriptors': feats[b][valid[b]]} for b in range(B)
			   ]

	@torch.inference_mode()
	def detectAndComputeDense(self, x, top_k = None, multiscale = True):
		"""
			Compute dense *and coarse* descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return: features sorted by their reliability score -- from most to least
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
					'scales'       ->   torch.Tensor(top_k,): extraction scale
					'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
		"""
		"""
		计算密集且粗粒度的描述符（支持批量处理）。

		输入:
			x -> torch.Tensor(B, C, H, W): 灰度或RGB图像（Batch, 通道, 高度, 宽度）
			top_k -> int: 保留的最佳特征点数量（若为None则使用类默认值self.top_k）
			multiscale -> bool: 是否启用多尺度特征提取（默认为True）

		返回: 按置信度排序的特征（从高到低）
			List[Dict]: 每个样本的字典包含:
				'keypoints'   -> torch.Tensor(top_k, 2): 粗粒度关键点坐标(x,y)
				'scales'      -> torch.Tensor(top_k,): 特征点提取时的尺度（仅多尺度模式下有效）
				'descriptors' -> torch.Tensor(top_k, 64): 粗粒度局部特征描述符
		"""
		# 1. 处理top_k参数：若未指定则使用类成员变量self.top_k
		if top_k is None: top_k = self.top_k
		# 2. 多尺度与单尺度分支选择
		if multiscale:
			# 多尺度模式：调用双尺度特征提取方法
			mkpts, sc, feats = self.extract_dualscale(x, top_k)  # 返回关键点、尺度和描述符
		else:
			# 单尺度模式：调用密集特征提取方法
			mkpts, feats = self.extractDense(x, top_k) # 返回关键点和描述符
			sc = torch.ones(mkpts.shape[:2], device=mkpts.device) # 生成默认尺度值（全1）

		return {'keypoints': mkpts,
				'descriptors': feats,
				'scales': sc }

	# 将XFeat提取的关键点、描述符和图像尺寸转换为LightGlue要求的格式，自动处理设备（CPU/GPU）和维度（添加batch维度）
	# 使用LightGlue算法（kornia库实现）计算特征对应关系,根据min_conf阈值过滤低质量匹配（默认保留置信度>0.1的匹配）
	@torch.inference_mode()
	def match_lighterglue(self, d0, d1, min_conf = 0.1):
		"""
			Match XFeat sparse features with LightGlue (smaller version) -- currently does NOT support batched inference because of padding, but its possible to implement easily.
			input:
				d0, d1: Dict('keypoints', 'scores, 'descriptors', 'image_size (Width, Height)')
			output:
				mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
                                idx              -> np.ndarray (N,2) the indices of the matching features
				
		"""
		"""
		使用LightGlue（轻量版）匹配XFeat稀疏特征。
		注：当前不支持批量推理（因padding问题），但易于扩展实现。

		输入:
			d0, d1: 字典，包含以下键:
				'keypoints': 关键点坐标 [N,2] (x,y)
				'scores': 关键点置信度 [N,]
				'descriptors': 特征描述符 [N, D] (D为特征维度)
				'image_size': 图像尺寸 (Width, Height)
			min_conf: 最低匹配置信度阈值（默认0.1）

		输出:
			mkpts_0, mkpts_1 -> np.ndarray [N,2]: 图像1到图像2的匹配关键点坐标
			idx              -> np.ndarray [N,2]: 匹配特征的索引对
		"""
		# 1. 检查依赖库kornia是否可用
		if not self.kornia_available:
			raise RuntimeError('We rely on kornia for LightGlue. Install with: pip install kornia')

		# 2. 惰性初始化LightGlue匹配器
		elif self.lighterglue is None:
			from modules.lighterglue import LighterGlue
			self.lighterglue = LighterGlue() # 实例化轻量级匹配器

		# 3. 构建LightGlue所需的输入数据格式
		data = {
				'keypoints0': d0['keypoints'][None, ...], # 增加batch维度 [1, N, 2]
				'keypoints1': d1['keypoints'][None, ...],
				'descriptors0': d0['descriptors'][None, ...],# [1, N, D]
				'descriptors1': d1['descriptors'][None, ...],
				'image_size0': torch.tensor(d0['image_size']).to(self.dev)[None, ...], # 图像尺寸转为tensor [1, 2]
				'image_size1': torch.tensor(d1['image_size']).to(self.dev)[None, ...]
		}

		#Dict -> log_assignment: [B x M+1 x N+1] matches0: [B x M] matching_scores0: [B x M] matches1: [B x N] matching_scores1: [B x N] matches: List[[Si x 2]], scores: List[[Si]]
		# 4. 执行特征匹配
		# 输出字典包含:
		#   matches: 匹配索引对列表 [Si, 2] (Si为有效匹配数)
		#   scores: 匹配得分列表 [Si]
		out = self.lighterglue(data, min_conf = min_conf)

		# 5. 提取匹配结果
		idxs = out['matches'][0] # 获取第一个（也是唯一一个）batch的匹配对

		return d0['keypoints'][idxs[:, 0]].cpu().numpy(), d1['keypoints'][idxs[:, 1]].cpu().numpy(), out['matches'][0].cpu().numpy()


	@torch.inference_mode()
	def match_xfeat(self, img1, img2, top_k = None, min_cossim = -1):
		"""
			Simple extractor and MNN matcher.
			简单的特征提取器+最近邻匹配器(MNN)。
			For simplicity it does not support batched mode due to possibly different number of kpts.
			为了简单起见，目前不支持批量模式，因为可能存在不同数量的特征点。
			input:
				img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				top_k -> int: keep best k features
			returns:
				mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
		"""
		if top_k is None: top_k = self.top_k
		img1 = self.parse_input(img1)
		img2 = self.parse_input(img2)

		# 3. 提取特征和关键点
		out1 = self.detectAndCompute(img1, top_k=top_k)[0]
		out2 = self.detectAndCompute(img2, top_k=top_k)[0]

		idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim )

		return out1['keypoints'][idxs0].cpu().numpy(), out2['keypoints'][idxs1].cpu().numpy()

	@torch.inference_mode()
	def match_xfeat_star(self, im_set1, im_set2, top_k = None):
		"""
			Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
			提取粗粒度特征 -> 批量匹配 -> 精细化匹配，当前支持批量模式。
			input:
				im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				top_k -> int: keep best k features
			returns:
				matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
				批量匹配结果列表，每个元素为匹配对(x1,y1,x2,y2) 或当B=1时返回元组(np.ndarray, np.ndarray)
		"""
		if top_k is None: top_k = self.top_k
		im_set1 = self.parse_input(im_set1)
		im_set2 = self.parse_input(im_set2)

		#Compute coarse feats
		# 3. 提取粗粒度特征（支持批量）
		out1 = self.detectAndComputeDense(im_set1, top_k=top_k)
		out2 = self.detectAndComputeDense(im_set2, top_k=top_k)

		#Match batches of pairs
		# 4. 批量匹配特征描述符
		idxs_list = self.batch_match(out1['descriptors'], out2['descriptors'] )
		B = len(im_set1)

		#Refine coarse matches
		#this part is harder to batch, currently iterate
		matches = []
		for b in range(B):
			matches.append(self.refine_matches(out1, out2, matches = idxs_list, batch_idx=b))

		return matches if B > 1 else (matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy())

	def preprocess_tensor(self, x):
		""" Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
		""" 
		确保图像尺寸能被32整除，避免混叠伪影。
		处理流程：
		1. 统一输入格式 -> torch.Tensor(B,C,H,W)
		2. 调整尺寸至32的倍数
		3. 返回处理后张量及缩放比例
		"""
		# 1. 处理numpy数组输入（非PyTorch Tensor情况）
		if isinstance(x, np.ndarray):
			if len(x.shape) == 3:
				x = torch.tensor(x).permute(2,0,1)[None]
			elif len(x.shape) == 2:
				x = torch.tensor(x[..., None]).permute(2,0,1)[None]
			else:
				raise RuntimeError('For numpy arrays, only (H,W) or (H,W,C) format is supported.')

		# 2. 校验张量维度（必须为4D: B,C,H,W）
		if len(x.shape) != 4:
			raise RuntimeError('Input tensor needs to be in (B,C,H,W) format')
	
		x = x.to(self.dev).float()  # 转移到指定设备并转为float32

		# 4. 计算调整后的尺寸（向下取整到最近的32倍数）
		H, W = x.shape[-2:] # 获取原始高宽
		_H, _W = (H//32) * 32, (W//32) * 32 # 新尺寸
		rh, rw = H/_H, W/_W

		# 5. 执行双线性插值调整尺寸
		x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
		return x, rh, rw

	def get_kpts_heatmap(self, kpts, softmax_temp = 1.0):
		"""
		    将关键点置信度图转换为高分辨率热图

		    输入:
		        kpts -> torch.Tensor(B, C, H, W): 原始关键点置信度图
		        softmax_temp -> float: 控制softmax分布尖锐度的温度参数

		    返回:
		        heatmap -> torch.Tensor(B, 1, H*8, W*8): 放大8倍后的热力图
		    """
		# 1. 对原始置信度图进行softmax归一化（按通道维度）
		# softmax_temp控制分布尖锐度（值越大峰值越突出）
		scores = F.softmax(kpts*softmax_temp, 1)[:, :64] # 只保留前64个通道
		# scores形状: [B, 64, H, W]

		# 2. 获取输入张量的基本维度信息
		B, _, H, W = scores.shape
		# 3. 通道重组构建高分辨率热图
		# 步骤分解：
		#   a) 将64通道重排为8x8的空间块
		#   b) 通过reshape和permute实现空间上采样
		heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
		heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
		return heatmap

	def NMS(self, x, threshold = 0.05, kernel_size = 5):
		"""
		非极大值抑制(Non-Maximum Suppression)实现
		功能：在热力图上定位局部最大值点（关键点候选）

		输入:
			x -> torch.Tensor(B, 1, H, W): 输入热力图（值越高表示关键点概率越大）
			threshold -> float: 响应值阈值（默认0.05）
			kernel_size -> int: 局部最大检测的邻域大小（默认5x5）

		返回:
			pos -> torch.Tensor(B, N, 2): 关键点坐标（每个样本最多N个点，不足处补0）
		"""
		# 1. 获取输入维度信息
		B, _, H, W = x.shape
		# 2. 计算局部最大值图
		pad=kernel_size//2     # 计算需要的padding量（保持尺寸不变）
		local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x) # 输出形状[B,1,H,W]
		pos = (x == local_max) & (x > threshold)
		pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

		# 3. 定位关键点候选
		# 条件1：当前像素是局部最大值
		# 条件2：响应值超过阈值
		pad_val = max([len(x) for x in pos_batched])

		# 4. 转换坐标格式
		pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

		#Pad kpts and build (B, N, 2) tensor
		# 6. 填充实际关键点（不足部分补0）
		for b in range(len(pos_batched)):
			pos[b, :len(pos_batched[b]), :] = pos_batched[b]

		return pos

	@torch.inference_mode()
	def batch_match(self, feats1, feats2, min_cossim = -1):
		"""
		批量双向最近邻特征匹配
		输入:
			feats1: [B,N,D] 图像组1的特征描述符（Batch, 点数, 维度）
			feats2: [B,M,D] 图像组2的特征描述符
			min_cossim: 最低余弦相似度阈值（-1表示不过滤）
		输出:
			List[Tuple]: 每个样本的匹配索引对(idx0, idx1)
		"""
		# 1. 获取批量大小
		B = len(feats1)
		# 2. 计算批量余弦相似度矩阵 [B,N,M]
		# feats2.permute(0,2,1)将特征转为[B,D,M]
		# bmm执行批量矩阵乘法得到相似度矩阵
		cossim = torch.bmm(feats1, feats2.permute(0,2,1))
		match12 = torch.argmax(cossim, dim=-1)
		match21 = torch.argmax(cossim.permute(0,2,1), dim=-1)

		idx0 = torch.arange(len(match12[0]), device=match12.device)
		# 5. 批量处理每个样本
		batched_matches = []

		for b in range(B):
			# 获取每个点的最大相似度值
			mutual = match21[b][match12[b]] == idx0

			if min_cossim > 0:
				cossim_max, _ = cossim[b].max(dim=1)
				good = cossim_max > min_cossim
				idx0_b = idx0[mutual & good]
				idx1_b = match12[b][mutual & good]
			else:
				idx0_b = idx0[mutual]
				idx1_b = match12[b][mutual]

			batched_matches.append((idx0_b, idx1_b))

		return batched_matches

	def subpix_softmax2d(self, heatmaps, temp = 3):
		"""
		使用softmax加权平均计算亚像素级坐标偏移量
		输入:
			heatmaps: [N, H, W] 局部热力图（N个点，每个点对应HxW的响应图）
			temp: softmax温度系数（值越大峰值越尖锐）
		输出:
			coords: [N, 2] 亚像素级偏移量(x,y)
		"""
		# 获取输入维度
		N, H, W = heatmaps.shape
		# 对热力图进行softmax归一化（保持局部窗口的响应概率）
		heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)

		# 生成网格坐标（中心零点化）
		x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
		x = x - (W//2)
		y = y - (H//2)

		# 计算加权偏移量（通过响应值加权平均坐标）
		coords_x = (x[None, ...] * heatmaps)
		coords_y = (y[None, ...] * heatmaps)
		coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
		coords = coords.sum(1)

		return coords

	def refine_matches(self, d0, d1, matches, batch_idx, fine_conf = 0.25):
		"""
		精细化匹配关键点坐标
		输入:
			d0/d1: 包含特征数据的字典（keypoints/descriptors/scales）
			matches: 初步匹配结果列表
			batch_idx: 当前处理的批次索引
			fine_conf: 置信度阈值
		输出:
			[N,4] 精细化后的匹配对(x1,y1,x2,y2)
		"""
		idx0, idx1 = matches[batch_idx]
		feats1 = d0['descriptors'][batch_idx][idx0]
		feats2 = d1['descriptors'][batch_idx][idx1]
		mkpts_0 = d0['keypoints'][batch_idx][idx0]
		mkpts_1 = d1['keypoints'][batch_idx][idx1]
		sc0 = d0['scales'][batch_idx][idx0]

		#Compute fine offsets
		# 3. 计算精细化偏移量
		# 3.1 通过小型网络预测偏移量（输入为拼接的特征对）
		offsets = self.net.fine_matcher(torch.cat([feats1, feats2],dim=-1))
		# 3.2 计算匹配置信度
		conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
		# 3.3 亚像素级偏移量计算
		offsets = self.subpix_softmax2d(offsets.view(-1,8,8))
		# 4. 应用偏移（考虑特征尺度）
		mkpts_0 += offsets* (sc0[:,None]) #*0.9 #* (sc0[:,None])
		# 5. 置信度过滤
		mask_good = conf > fine_conf
		mkpts_0 = mkpts_0[mask_good]
		mkpts_1 = mkpts_1[mask_good]

		return torch.cat([mkpts_0, mkpts_1], dim=-1)

	@torch.inference_mode()
	def match(self, feats1, feats2, min_cossim = 0.82):
		"""
		双向最近邻特征匹配（单样本版）

		输入:
			feats1: [N,D] 第一幅图像的特征描述符（N个点，D维特征）
			feats2: [M,D] 第二幅图像的特征描述符
			min_cossim: 最低余弦相似度阈值（默认0.82）

		输出:
			idx0: 第一幅图像的匹配点索引
			idx1: 第二幅图像的匹配点索引
		"""

		# 1. 计算余弦相似度矩阵
		# feats1 @ feats2.t() 等价于 torch.matmul(feats1, feats2.transpose(0,1))
		cossim = feats1 @ feats2.t()
		cossim_t = feats2 @ feats1.t()
		
		_, match12 = cossim.max(dim=1)
		_, match21 = cossim_t.max(dim=1)

		idx0 = torch.arange(len(match12), device=match12.device)
		mutual = match21[match12] == idx0

		# 5. 相似度阈值过滤
		if min_cossim > 0:
			cossim, _ = cossim.max(dim=1)
			good = cossim > min_cossim
			idx0 = idx0[mutual & good]
			idx1 = match12[mutual & good]
		else:
			idx0 = idx0[mutual]
			idx1 = match12[mutual]

		return idx0, idx1

	def create_xy(self, h, w, dev):
		"""
		生成网格坐标点
		输入:
			h: 高度
			w: 宽度
			dev: 设备(CPU/GPU)
		输出:
			xy: [h*w, 2] 的坐标矩阵，每行是(x,y)坐标
		"""
		y, x = torch.meshgrid(torch.arange(h, device = dev), 
								torch.arange(w, device = dev), indexing='ij')
		xy = torch.cat([x[..., None],y[..., None]], -1).reshape(-1,2)
		return xy

	def extractDense(self, x, top_k = 8_000):
		"""
		密集特征提取方法
		输入:
			x: 输入图像 [B,C,H,W]
			top_k: 要保留的top特征点数
		输出:
			mkpts: 关键点坐标 [B,top_k,2] (x,y)
			feats: 特征描述符 [B,top_k,64]
		"""
		if top_k < 1:
			top_k = 100_000_000

		x, rh1, rw1 = self.preprocess_tensor(x)

		M1, K1, H1 = self.net(x)
		
		B, C, _H1, _W1 = M1.shape
		
		xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B,-1,-1)

		M1 = M1.permute(0,2,3,1).reshape(B, -1, C)
		H1 = H1.permute(0,2,3,1).reshape(B, -1)

		_, top_k = torch.topk(H1, k = min(len(H1[0]), top_k), dim=-1)

		feats = torch.gather( M1, 1, top_k[...,None].expand(-1, -1, 64))
		mkpts = torch.gather(xy1, 1, top_k[...,None].expand(-1, -1, 2))
		mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1,-1)

		return mkpts, feats

	def extract_dualscale(self, x, top_k, s1 = 0.6, s2 = 1.3):
		"""
		双尺度特征提取方法
		输入:
			x: 输入图像 [B,C,H,W]
			top_k: 要保留的总特征点数
			s1: 第一尺度下采样因子（默认0.6）
			s2: 第二尺度上采样因子（默认1.3）
		输出:
			mkpts: 关键点坐标 [B,top_k,2] (x,y)
			sc: 特征尺度 [B,top_k]
			feats: 特征描述符 [B,top_k,D]
		"""
		x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode='bilinear')
		x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode='bilinear')

		B, _, _, _ = x.shape

		mkpts_1, feats_1 = self.extractDense(x1, int(top_k*0.20))
		mkpts_2, feats_2 = self.extractDense(x2, int(top_k*0.80))

		mkpts = torch.cat([mkpts_1/s1, mkpts_2/s2], dim=1)
		sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1/s1)
		sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1/s2)
		sc = torch.cat([sc1, sc2],dim=1)
		feats = torch.cat([feats_1, feats_2], dim=1)

		return mkpts, sc, feats

	def parse_input(self, x):
		"""
		输入数据标准化处理
		输入:
			x: 可以是np.ndarray或torch.Tensor
				- 形状为 (H,W,C) 或 (B,H,W,C) 的numpy数组
				- 形状为 (B,C,H,W) 的torch张量
		输出:
			标准化的torch.Tensor [B,C,H,W]，值域[0,1]
		"""
		# 1. 处理单张图像输入（自动添加batch维度）
		if len(x.shape) == 3:
			x = x[None, ...]
		# 2. 处理numpy数组输入
		if isinstance(x, np.ndarray):
			x = torch.tensor(x).permute(0,3,1,2)/255

		return x
