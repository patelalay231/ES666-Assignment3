import pdb
import glob
import cv2
import os

import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm
import random

class PanaromaStitcher:
    def __init__(self):
        pass

    def show_image_grid(self, images, M=1, N=1, title='Title', figsize=8):
        # Display a grid of images, given as a numpy array of shape (num_images, height, width, channels)
        if M == 1:
            fig_height, fig_width = figsize, figsize // 4
        elif N == 1:
            fig_height, fig_width = figsize // 4, figsize
        else:
            fig_height, fig_width = figsize, figsize

        fig, axes = plt.subplots(M, N, figsize=(fig_height, fig_width))
        
        # Ensure images have 4 dimensions (num_images, height, width, channels)
        if len(images.shape) < 4:
            images = np.expand_dims(images.copy(), axis=0)

        for i in range(M):
            for j in range(N):
                ax = axes if M == 1 and N == 1 else (axes[max(i, j)] if M == 1 or N == 1 else axes[i, j])
                img_idx = i * N + j
                if img_idx < images.shape[0]:
                    ax.imshow(cv2.cvtColor(images[img_idx], cv2.COLOR_BGR2RGB))
                ax.axis('off')
        plt.tight_layout()
        plt.show()

    def feature_detector(self, image1, image2, algo="blob", nfeatures=10000):
        # Convert images to grayscale
        gray_img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        if algo == "blob":
            orb = cv2.ORB_create(nfeatures=nfeatures)
            keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
            keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

            bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf_matcher.match(descriptors1, descriptors2)

            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        else:
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

            bf_matcher = cv2.BFMatcher()
            matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)

            good_matches = []
            for m in matches:
                if m[0].distance < 0.5 * m[1].distance:
                    good_matches.append(m)
            matches = np.asarray(good_matches)

            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)

        return src_pts, dst_pts

    def compute_homography(self, pointR, pointL):
        # Ensure sufficient points are available for homography computation
        if len(pointL) < 4:
            raise ValueError("Insufficient points for homography")

        matrix_a = []
        for i in range(len(pointL)):
            x, y = pointR[i][0], pointR[i][1]
            x_prime, y_prime = pointL[i][0], pointL[i][1]

            matrix_a.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
            matrix_a.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])

        U, S, Vt = np.linalg.svd(np.array(matrix_a))
        H_matrix = Vt[-1].reshape(3, 3)
        H_matrix /= H_matrix[2, 2]
        return H_matrix

    def ransac(self, src_points, dst_points, point_selection=4, iterations=10000, threshold=4.0):
        best_homography = None
        max_inliers_count = 0

        for _ in range(iterations):
            sample_indices = random.sample(range(len(src_points)), point_selection)
            src_sample, dst_sample = src_points[sample_indices], dst_points[sample_indices]

            H_candidate = self.compute_homography(src_sample, dst_sample)

            # Evaluate homography using inliers
            src_homogeneous = np.hstack((src_points, np.ones((len(src_points), 1))))
            projected_points = (H_candidate @ src_homogeneous.T).T
            projected_points /= projected_points[:, 2].reshape(-1, 1)

            distances = np.sqrt(np.sum((projected_points[:, :2] - dst_points) ** 2, axis=1))
            inliers = distances < threshold
            inliers_count = np.sum(inliers)

            if inliers_count > max_inliers_count:
                max_inliers_count = inliers_count
                best_homography = H_candidate

        return best_homography, max_inliers_count

    def warpPerspective(self, imageR, imageL, size_x, size_y, H):
        # Transform imageR to align with imageL using homography matrix H
        h, w = size_y, size_x
        output_img = np.zeros((size_y, size_x, 3), dtype=imageR.dtype)
        H_inv = np.linalg.inv(H)

        for x in range(size_x):
            for y in range(size_y):
                coord = np.array([x, y, 1], dtype=np.float32)
                transformed_coord = H_inv @ coord
                transformed_coord /= transformed_coord[2]

                src_x, src_y = int(transformed_coord[0]), int(transformed_coord[1])
                if 0 <= src_x < imageR.shape[1] and 0 <= src_y < imageR.shape[0]:
                    output_img[y, x] = imageR[src_y, src_x]

        return output_img

    def remove_Pad(self, image):
        # Crop padding from the image
        rows = np.any(image != 0, axis=1)
        cols = np.any(image != 0, axis=0)

        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        return image[row_min:row_max + 1, col_min:col_max + 1]

    def make_panaroma_for_images_in(self, path):
        print("Path:", path)
        img_folder = path
        images_found = sorted(glob.glob(img_folder + os.sep + '*'))
        print(f'Found {len(images_found)} images for stitching.')

        gt_images = [cv2.imread(img_path, 1) for img_path in tqdm(natsorted(glob.glob(f"{path}/*")))]
        gt_images_resized = [np.array(gt_images, dtype=object)]

        homography_matrices = []
        img_idx = 0

        temp_image = gt_images_resized[img_idx][-1]
        temp_image = cv2.resize(temp_image.astype(np.uint8), (640, 480))

        for i in range(len(gt_images_resized[img_idx]) - 2, -1, -1):
            right_img = temp_image.astype(np.uint8)
            left_img = gt_images_resized[img_idx][i].astype(np.uint8)
            left_img = cv2.resize(left_img, (640, 480))

            points_left, points_right = self.feature_detector(left_img, right_img)

            retry_counter = 0
            while True:
                H_matrix, inlier_count = self.ransac(points_right.copy(), points_left.copy())
                if inlier_count >= 200 or retry_counter > 10:
                    break
                retry_counter += 1

            if inlier_count < 200:
                break

            homography_matrices.append(H_matrix)

            warped_image = self.warpPerspective(right_img, left_img, (right_img.shape[1] + left_img.shape[1]),
                                                100 + max(right_img.shape[0], left_img.shape[0]), H_matrix)

            new_output_image = warped_image.copy()
            new_output_image[:left_img.shape[0], :left_img.shape[1]] = left_img
            temp_image = self.remove_Pad(new_output_image)

        return temp_image, homography_matrices

    def say_hi(self):
        raise NotImplementedError("This method is not yet implemented.")
