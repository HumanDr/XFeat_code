import os
import cv2
import numpy as np
from modules.xfeat import XFeat
import torch
import time
import matplotlib.pyplot as plt


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
    def __init__(self, ref_image_path, target_image_path, width=640, height=480, max_kpts=3000):
        self.ref_frame = cv2.imread(ref_image_path)  # Reference image
        self.current_frame = cv2.imread(target_image_path)  # Target image

        if self.ref_frame is None or self.current_frame is None:
            print("Error: One or both of the images failed to load.")
            exit()

        # Resize images to fit for display if needed (optional)
        # self.ref_frame = cv2.resize(self.ref_frame, (width, height))
        # self.current_frame = cv2.resize(self.current_frame, (width, height))

        # Initialize XFeat method
        self.method = Method(descriptor=CVWrapper(XFeat(top_k=max_kpts)), matcher=XFeat())

        # Precompute features for reference frame
        self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None)

        # Homography params
        self.min_inliers = 50
        self.ransac_thr = 4.0

        # Setup text font and other UI elements
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.9
        self.line_type = cv2.LINE_AA
        self.line_color = (0, 255, 0)  # Default line color
        self.line_thickness = 4

        self.window_name = "Image Matching - Press 's' to update reference."

    def putText(self, canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
        # Draw the border
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale,
                    color=borderColor, thickness=thickness + 2, lineType=lineType)
        # Draw the text
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale,
                    color=textColor, thickness=thickness, lineType=lineType)

    def match_and_draw(self, ref_frame, current_frame):
        matches, good_matches = [], []
        kp1, kp2 = [], []
        points1, points2 = [], []

        # Detect and compute features using XFeat
        current = self.method.descriptor.detectAndCompute(current_frame)
        kpts1, descs1 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']
        kpts2, descs2 = current['keypoints'], current['descriptors']

        # Use XFeat matcher to find matches
        idx0, idx1 = self.method.matcher.match(descs1, descs2, 0.82)
        points1 = kpts1[idx0].cpu().numpy()
        points2 = kpts2[idx1].cpu().numpy()

        if len(points1) > 10 and len(points2) > 10:
            # Find homography
            H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=700,
                                            confidence=0.995)
            inliers = inliers.flatten() > 0

            if inliers.sum() < self.min_inliers:
                H = None

            kp1 = [cv2.KeyPoint(p[0], p[1], 5) for p in points1[inliers]]
            kp2 = [cv2.KeyPoint(p[0], p[1], 5) for p in points2[inliers]]
            good_matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

            # Draw matches
            matched_frame = cv2.drawMatches(ref_frame, kp1, current_frame, kp2, good_matches, None,
                                            matchColor=(0, 200, 0), flags=2)
        else:
            matched_frame = np.hstack([ref_frame, current_frame])

        color = (240, 89, 169)  # Set color for text and border

        # Add a colored rectangle to separate from the top frame
        cv2.rectangle(matched_frame, (2, 2), (self.ref_frame.shape[1] * 2 - 2, self.ref_frame.shape[0] - 2), color, 5)

        # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="XFeat Matches: %d" % len(good_matches), org=(10, 30),
                     fontFace=self.font,
                     fontScale=self.font_scale, textColor=(0, 0, 0), borderColor=color, thickness=1,
                     lineType=self.line_type)

        return matched_frame

    def process(self):
        # Start timer for matching process
        start_time = time.time()

        # Match features and draw matches on the result frame
        result_frame = self.match_and_draw(self.ref_frame, self.current_frame)

        # Calculate the time taken for matching
        elapsed_time = time.time() - start_time

        # Add the elapsed time as text on the result image
        color = (240, 89, 169)  # Set color for time text
        self.putText(canvas=result_frame, text="Time: %.4f seconds" % elapsed_time,
                     org=(10, self.ref_frame.shape[0] - 10),
                     fontFace=self.font, fontScale=self.font_scale, textColor=(0, 0, 0), borderColor=color, thickness=1,
                     lineType=self.line_type)

        # Display the result using plt

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axes
        plt.show()


    def main_loop(self):
        self.process()


if __name__ == "__main__":
    # Hardcoded image paths
    ref_image_path = 'assets/cur_denoise/1.png'  # Replace with your reference image path
    target_image_path = 'assets/cur_denoise/2.png'  # Replace with your target image path

    # Initialize and run the demo
    demo = MatchingDemo(ref_image_path=ref_image_path, target_image_path=target_image_path)
    demo.main_loop()
