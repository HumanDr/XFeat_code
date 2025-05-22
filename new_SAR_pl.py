import os
import cv2
import numpy as np
from modules.xfeat import XFeat
import torch
import time
import matplotlib.pyplot as plt
from datetime import datetime
from natsort import natsorted  # 用于自然排序文件名


class CVWrapper():
    def __init__(self, mtd):
        self.mtd = mtd

    def detectAndCompute(self, x, mask=None):
        return self.mtd.detectAndCompute(torch.tensor(x).permute(2, 0, 1).float()[None])[0]


class Method:
    def __init__(self, descriptor, matcher):
        self.descriptor = descriptor
        self.matcher = matcher


class BatchMatchingDemo:
    def __init__(self, image_dir, output_dir="batch_match_results", max_kpts=3000):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.max_kpts = max_kpts

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 获取并排序图像文件
        self.image_files = self._get_image_files()
        if len(self.image_files) < 2:
            print("错误: 需要至少2张图像进行匹配")
            exit()

        # 初始化XFeat方法
        self.method = Method(descriptor=CVWrapper(XFeat(top_k=max_kpts)), matcher=XFeat())

        # 单应性参数
        self.min_inliers = 50
        self.ransac_thr = 4.0

        # 设置文本样式
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.9
        self.line_type = cv2.LINE_AA
        self.line_thickness = 2

    def _get_image_files(self):
        """获取目录中的所有图像文件并自然排序"""
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        files = [f for f in os.listdir(self.image_dir)
                 if f.lower().endswith(valid_ext)]
        return natsorted(files)  # 自然排序(1,2,10而不是1,10,2)

    def _load_image(self, filename):
        """加载图像并确保是三通道"""
        img = cv2.imread(os.path.join(self.image_dir, filename))
        if img is None:
            print(f"警告: 无法加载图像 {filename}")
            return None
        return img

    def match_pair(self, img1, img2):
        """匹配一对图像并返回结果"""
        # 检测和计算特征
        kpts1, descs1 = self.method.descriptor.detectAndCompute(img1)['keypoints'], \
            self.method.descriptor.detectAndCompute(img1)['descriptors']
        kpts2, descs2 = self.method.descriptor.detectAndCompute(img2)['keypoints'], \
            self.method.descriptor.detectAndCompute(img2)['descriptors']

        # 匹配特征
        idx0, idx1 = self.method.matcher.match(descs1, descs2, 0.82)
        points1 = kpts1[idx0].cpu().numpy()
        points2 = kpts2[idx1].cpu().numpy()

        # 准备可视化画布（带间隙）
        h, w = img1.shape[:2]
        gap = 50  # 间隙宽度
        canvas = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
        canvas[:, :w] = img1
        canvas[:, -w:] = img2

        if len(points1) > 10 and len(points2) > 10:
            # 计算单应性
            H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC,
                                            self.ransac_thr, maxIters=700, confidence=0.995)
            inliers = inliers.flatten() > 0

            if inliers.sum() >= self.min_inliers:
                # 调整右图坐标以考虑间隙
                points2[:, 0] += w + gap

                # 绘制匹配线和特征点
                for p1, p2 in zip(points1[inliers], points2[inliers]):
                    cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)),
                             (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(canvas, tuple(p1.astype(int)), 3, (0, 255, 0), -1)
                    cv2.circle(canvas, tuple(p2.astype(int)), 3, (0, 255, 0), -1)

                match_count = inliers.sum()
            else:
                match_count = 0
        else:
            match_count = 0

        # 添加文字信息
        text_color = (0, 0, 255)  # 红色文字
        cv2.putText(canvas, f"Matches: {match_count}", (10, 30),
                    self.font, self.font_scale, text_color, self.line_thickness, self.line_type)

        return canvas

    def process_all_pairs(self):
        """处理所有连续的图像对"""
        print(f"开始批量处理 {len(self.image_files)} 张图像...")

        for i in range(len(self.image_files) - 1):
            start_time = time.time()

            # 加载图像对
            img1 = self._load_image(self.image_files[i])
            img2 = self._load_image(self.image_files[i + 1])
            if img1 is None or img2 is None:
                continue

            # 执行匹配
            result = self.match_pair(img1, img2)
            # 计算耗时
            elapsed = time.time() - start_time
            text_color = (0, 0, 255)
            cv2.putText(result, f"Time: {elapsed:.4f}seconds", (10, result.shape[0] - 10),
                        self.font, self.font_scale, text_color, self.line_thickness, self.line_type)

            # 保存结果
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"match_{i + 1}-{i + 2}.png"
            output_path = os.path.join(self.output_dir, output_name)
            cv2.imwrite(output_path, result)


            print(f"已处理 {self.image_files[i]} ↔ {self.image_files[i + 1]} - "
                  f"匹配点: {int(result.mean())} - 耗时: {elapsed:.2f}s")

            # 显示进度
            # if i < len(self.image_files) - 2:  # 不是最后一对时显示
            #     plt.figure(figsize=(12, 6))
            #     plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            #     plt.title(f"Matching {i + 1}/{len(self.image_files) - 1}")
            #     plt.axis('off')
            #     plt.show(block=False)
            #     plt.pause(1)
            #     plt.close()

        print(f"处理完成! 结果已保存到 {self.output_dir}")


if __name__ == "__main__":
    # 配置路径
    image_directory = "assets/0608PNG"  # 包含待匹配图像的文件夹
    output_directory = "batch_results_0608(1)"  # 结果输出文件夹

    # 初始化并运行批量匹配
    matcher = BatchMatchingDemo(image_dir=image_directory,
                                output_dir=output_directory)
    matcher.process_all_pairs()