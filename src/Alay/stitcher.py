import pdb
import glob
import cv2
import os
import importlib
import random
import numpy as np
import matplotlib.pyplot as plt

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} Images for stitching')
        homographies_list = []
        final_image = cv2.imread(all_images[0])  # Load the first image
        # Iterate through images and stitch each one to the previously stitched result
        for idx in range(1, len(all_images)):
            next_image = cv2.imread(all_images[idx])
            final_image, homography = self.Stitch(final_image, next_image)
            homographies_list.append(homography)
        return final_image, homographies_list

    def Stitch(self, img_left, img_right):
        kp1, desc1, kp2, desc2 = self.Keypoints(img_left, img_right)  # Find keypoints and descriptors for both images
        matches = self.Match_Keypoints(kp1, kp2, desc1, desc2)  # Match keypoints between images
        H_matrix = self.estimate_homography(matches)  # Estimate homography matrix with RANSAC
        rows_r, cols_r = img_right.shape[:2]
        rows_l, cols_l = img_left.shape[:2]

        # Map corner points using the homography
        corners_r = np.float32([[0, 0], [0, rows_r], [cols_r, rows_r], [cols_r, 0]]).reshape(-1, 1, 2)
        corners_l = np.float32([[0, 0], [0, rows_l], [cols_l, rows_l], [cols_l, 0]]).reshape(-1, 1, 2)
        transformed_corners_l = self.apply_homography(corners_l, H_matrix)
        all_corners = np.concatenate((corners_r, transformed_corners_l), axis=0)

        # Determine final image size to accommodate both images
        [min_x, min_y] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [max_x, max_y] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translation_H = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]]) @ H_matrix  # Translation homography

        warped_img = self.warp_image(img_left, translation_H, (max_y - min_y, max_x - min_x))  # Warp left image

        # Place right image on the warped output
        offset_y = -min_y
        offset_x = -min_x
        warped_img[offset_y:offset_y + rows_r, offset_x:offset_x + cols_r] = img_right
        return warped_img, H_matrix

    def Keypoints(self, img_left, img_right):
        left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(left_gray, None)
        kp2, desc2 = sift.detectAndCompute(right_gray, None)
        return kp1, desc1, kp2, desc2

    def Match_Keypoints(self, kp1, kp2, desc1, desc2):
        matcher = cv2.BFMatcher()
        knn_matches = matcher.knnMatch(desc1, desc2, k=2)
        valid_matches = []
        for m, n in knn_matches:
            if m.distance < 0.75 * n.distance:
                pt_left = kp1[m.queryIdx].pt
                pt_right = kp2[m.trainIdx].pt
                valid_matches.append([pt_left[0], pt_left[1], pt_right[0], pt_right[1]])
        return valid_matches

    def estimate_homography(self, matches, num_iterations=5000, threshold=5):
        max_inliers = []
        best_H = None
        for _ in range(num_iterations):
            random_points = random.sample(matches, 4)
            H = self.compute_homography(random_points)
            inliers = []
            for match in matches:
                p_left = np.array([match[0], match[1], 1]).reshape(3, 1)
                p_right = np.array([match[2], match[3], 1]).reshape(3, 1)
                mapped_p = np.dot(H, p_left)
                mapped_p = mapped_p / mapped_p[2]
                dist = np.linalg.norm(p_right - mapped_p)
                if dist < threshold:
                    inliers.append(match)
            if len(inliers) > len(max_inliers):
                max_inliers = inliers
                best_H = H
        return best_H

    def compute_homography(self, point_pairs):
        matrix = []
        for pair in point_pairs:
            x1, y1 = pair[0], pair[1]
            x2, y2 = pair[2], pair[3]
            matrix.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
            matrix.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
        matrix = np.array(matrix)
        _, _, vh = np.linalg.svd(matrix)
        H = (vh[-1, :].reshape(3, 3))
        return H / H[2, 2]

    def apply_homography(self, corner_points, H):
        corner_points = np.array(corner_points, dtype=np.float32).reshape(-1, 1, 2)
        num_corners = corner_points.shape[0]
        homogeneous_points = np.concatenate([corner_points, np.ones((num_corners, 1, 1))], axis=-1)
        transformed_corners = np.dot(homogeneous_points, H.T)
        transformed_corners = transformed_corners[:, :, :2] / transformed_corners[:, :, 2:3]
        return transformed_corners.reshape(corner_points.shape)

    def warp_image(self, img, H, output_dims):
        H_inv = np.linalg.inv(H)
        height, width = output_dims
        warped = np.zeros((height, width, img.shape[2]), dtype=img.dtype)

        for y in range(height):
            for x in range(width):
                src = H_inv @ np.array([x, y, 1])
                src_x, src_y = src[:2] / src[2]
                if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                    x0, y0 = int(src_x), int(src_y)
                    x1, y1 = min(x0 + 1, img.shape[1] - 1), min(y0 + 1, img.shape[0] - 1)
                    dx, dy = src_x - x0, src_y - y0
                    warped[y, x] = ((1 - dx) * (1 - dy) * img[y0, x0] + dx * (1 - dy) * img[y0, x1] + (1 - dx) * dy * img[y1, x0] + dx * dy * img[y1, x1])
        return warped

    def say_hi(self):
        print("Hi, I'm a panorama stitcher!")