import os
import cv2
import numpy as np
from modules.xfeat import XFeat
import torch
import time
import matplotlib.pyplot as plt
from datetime import datetime


class CVWrapper():
    def __init__(self, mtd):
        self.mtd = mtd

    def detectAndCompute(self, x, mask=None):
        return self.mtd.detectAndCompute(torch.tensor(x).permute(2, 0, 1).float()[None])[0]


class Method:
    def __init__(self, descriptor, matcher):
        self.descriptor = descriptor
        self.matcher = matcher


class MatchingDemo:
    def __init__(self, ref_image_path, target_image_path, output_dir="output_results",
                 max_kpts=3000):
        self.ref_frame = cv2.imread(ref_image_path)  # 参考图像
        self.current_frame = cv2.imread(target_image_path)  # 目标图像
        self.output_dir = output_dir  # 输出目录

        # 创建输出目录（如果不存在）
        os.makedirs(self.output_dir, exist_ok=True)

        if self.ref_frame is None or self.current_frame is None:
            print("错误: 无法加载一张或两张图像。")
            exit()

        # 调整图像大小以适应显示（可选）
        # self.ref_frame = cv2.resize(self.ref_frame, (width, height))
        # self.current_frame = cv2.resize(self.current_frame, (width, height))

        # 初始化XFeat方法
        self.method = Method(descriptor=CVWrapper(XFeat(top_k=max_kpts)), matcher=XFeat())

        # 预计算参考帧的特征
        self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None)

        # 单应性参数
        self.min_inliers = 50
        self.ransac_thr = 2.0

        # 设置文本字体和其他UI元素
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.9
        self.line_type = cv2.LINE_AA
        self.line_color = (0, 255, 0)  # 默认线条颜色
        self.line_thickness = 4

        self.window_name = "图像匹配 - 按's'键更新参考图像。"

    def putText(self, canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
        # 绘制边框
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale,
                    color=borderColor, thickness=thickness + 2, lineType=lineType)
        # 绘制文本
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale,
                    color=textColor, thickness=thickness, lineType=lineType)

    def match_and_draw(self, ref_frame, current_frame):
        matches, good_matches = [], []
        kp1, kp2 = [], []
        points1, points2 = [], []

        # 使用XFeat检测和计算特征
        current = self.method.descriptor.detectAndCompute(current_frame)
        kpts1, descs1 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']
        kpts2, descs2 = current['keypoints'], current['descriptors']

        # 使用XFeat匹配器查找匹配
        idx0, idx1 = self.method.matcher.match(descs1, descs2, 0.82)
        points1 = kpts1[idx0].cpu().numpy()
        points2 = kpts2[idx1].cpu().numpy()

        if len(points1) > 10 and len(points2) > 10:
            # 查找单应性
            H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=700,
                                            confidence=0.995)
            inliers = inliers.flatten() > 0

            if inliers.sum() < self.min_inliers:
                H = None

            kp1 = [cv2.KeyPoint(p[0], p[1], 5) for p in points1[inliers]]
            kp2 = [cv2.KeyPoint(p[0], p[1], 5) for p in points2[inliers]]
            good_matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

            # 绘制匹配
            matched_frame = cv2.drawMatches(ref_frame, kp1, current_frame, kp2, good_matches, None,
                                            matchColor=(0, 200, 0), flags=2)
        else:
            matched_frame = np.hstack([ref_frame, current_frame])

        color = (240, 89, 169)  # 设置文本和边框的颜色

        # 添加彩色矩形以与顶部框架分隔
        # cv2.rectangle(matched_frame, (2, 2), (self.ref_frame.shape[1] * 2 - 2, self.ref_frame.shape[0] - 2), color, 5)

        # 在顶部框架画布上添加标题
        self.putText(canvas=matched_frame, text="XFeat Matches: %d" % len(good_matches), org=(10, 30),
                     fontFace=self.font,
                     fontScale=self.font_scale, textColor=(0, 0, 0), borderColor=color, thickness=1,
                     lineType=self.line_type)

        return matched_frame


    def save_result(self, result_frame):
        """保存结果图像到指定目录"""
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"match_result_{timestamp}.png"
        output_path = os.path.join(self.output_dir, filename)

        # 保存图像
        cv2.imwrite(output_path, result_frame)
        print(f"结果已保存到: {output_path}")

    def process(self):
        # 开始计时匹配过程
        start_time = time.time()

        # 匹配特征并在结果帧上绘制匹配
        result_frame = self.match_and_draw(self.ref_frame, self.current_frame)

        # 计算匹配所用时间
        elapsed_time = time.time() - start_time

        # 在结果图像上添加耗时文本
        color = (240, 89, 169)  # 时间文本的颜色
        self.putText(canvas=result_frame, text="Time: %.4f seconds" % elapsed_time,
                     org=(10, self.ref_frame.shape[0] - 10),
                     fontFace=self.font, fontScale=self.font_scale, textColor=(0, 0, 0), borderColor=color, thickness=1,
                     lineType=self.line_type)

        # 保存结果图像
        self.save_result(result_frame)

        # 使用plt显示结果
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # 隐藏坐标轴
        plt.show()

    def main_loop(self):
        self.process()


if __name__ == "__main__":
    # 硬编码的图像路径
    ref_image_path = 'assets/cur_denoise/10.png'  # 替换为你的参考图像路径
    target_image_path = 'assets/cur_denoise/11.png'  # 替换为你的目标图像路径

    # 初始化并运行演示
    demo = MatchingDemo(ref_image_path=ref_image_path,
                        target_image_path=target_image_path,
                        output_dir="match_results")  # 指定输出目录
    demo.main_loop()