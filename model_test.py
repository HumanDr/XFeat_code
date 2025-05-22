import numpy as np
import os
import torch
import torch.nn.functional as F

import tqdm

from modules.model import *
from modules.interpolator import InterpolateSparse2d

import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
def preprocess_image(image_path, target_size=256):
    """将图片转换为模型输入的张量"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])
    x = transform(img).unsqueeze(0).cuda()  # (1, 3, H, W)
    return x, img


# 可视化函数
def visualize_results(x, feats, keypoints, heatmap):
    """在2x2网格中显示结果"""
    plt.figure(figsize=(12, 12))

    # 子图1: 输入图像
    plt.subplot(2, 2, 1)
    plt.title("Input Image")
    plt.imshow(x.squeeze().permute(1, 2, 0).cpu())
    plt.axis('off')

    # 子图2: 特征图（取前3通道）
    plt.subplot(2, 2, 2)
    plt.title("Feature Maps (First 3 Channels)")
    # feats_vis = feats[0, :3].permute(1, 2, 0).cpu().detach()
    # feats_vis = (feats_vis - feats_vis.min()) / (feats_vis.max() - feats_vis.min())
    # plt.imshow(feats_vis)
    print(feats[0, 0].shape)
    print(feats[0,0])
    print(feats[0, 1])
    plt.imshow(feats[0, 0].detach().cpu())
    plt.axis('off')

    # 子图3: 关键点响应（64个网格的最大值）
    plt.subplot(2, 2, 3)
    plt.title("Keypoint Responses")
    kpts_vis = keypoints[0, :64].max(dim=0)[0].cpu().detach()
    plt.imshow(kpts_vis, cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # 子图4: 可靠性热图
    plt.subplot(2, 2, 4)
    plt.title("Reliability Heatmap")
    plt.imshow(heatmap[0, 0].cpu().detach(), cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    # 示例：处理一张本地图片
    image_path = "assets/R-C.jpg"  # 替换为你的图片路径
    x, original_img = preprocess_image(image_path)
    # x = torch.randn(1, 3, 256, 256).cuda()
    model = XFeatModel().eval().cuda()  # 切换到推理模式

    # 模型推理
    with torch.no_grad():
        feats, keypoints, heatmap = model(x)
        print(feats.shape,keypoints.shape,heatmap.shape)

    # 可视化
    visualize_results(x, feats, keypoints, heatmap)