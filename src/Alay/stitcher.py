import cv2
import numpy as np
import glob
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PanoramaLogger")

class PanaromaStitcher:
    def __init__(self):
        # Initialize SIFT detector and FLANN-based matcher
        self.sift = cv2.SIFT_create()
        flann_params = {'algorithm': 1, 'trees': 5}
        search_checks = {'checks': 50}
        self.matcher = cv2.FlannBasedMatcher(flann_params, search_checks)

    def make_panaroma_for_images_in(self, path):
        # Retrieve all images from the specified directory
        img_paths = glob.glob(f'{path}/*.*')
        img_list = [cv2.imread(p) for p in img_paths]

        # Begin with the first image as the base for stitching
        base_image = img_list[0]
        homographies = []

        for idx in range(1, len(img_list)):
            # Detect keypoints and compute descriptors for current images
            kp_base, des_base = self.sift.detectAndCompute(base_image, None)
            kp_next, des_next = self.sift.detectAndCompute(img_list[idx], None)

            if des_base is None or des_next is None:
                logger.warning(f"Descriptors missing in image {idx}. Skipping.")
                continue

            # Match features between images and apply ratio test
            matches = self.matcher.knnMatch(des_base, des_next, k=2)
            valid_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(valid_matches) < 6:
                logger.warning(f"Insufficient matches between image {idx} and image {idx - 1}. Skipping.")
                continue

            # Extract point coordinates from keypoints
            pts_base = np.float32([kp_base[m.queryIdx].pt for m in valid_matches]).reshape(-1, 2)
            pts_next = np.float32([kp_next[m.trainIdx].pt for m in valid_matches]).reshape(-1, 2)

            # Compute homography matrix with RANSAC
            homography = self.compute_homography_using_ransac(pts_next, pts_base)
            if homography is None:
                logger.warning(f"Failed to compute homography for image {idx} and image {idx - 1}. Skipping.")
                continue

            homographies.append(homography)
            base_image = self.inverse_warp(base_image, img_list[idx], homography)

        return base_image, homographies

    def normalize_points(self, points):
        # Calculate mean and standard deviation for normalization
        mean, std = np.mean(points, axis=0), np.std(points, axis=0)
        std[std < 1e-8] = 1e-8  # Prevent divide-by-zero errors
        scaling = np.sqrt(2) / std
        T = np.array([[scaling[0], 0, -scaling[0] * mean[0]],
                      [0, scaling[1], -scaling[1] * mean[1]],
                      [0, 0, 1]])
        pts_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        normalized_pts = (T @ pts_homogeneous.T).T
        return normalized_pts[:, :2], T

    def direct_linear_transform(self, pts1, pts2):
        pts1_norm, T1 = self.normalize_points(pts1)
        pts2_norm, T2 = self.normalize_points(pts2)
        A = []
        for pt1, pt2 in zip(pts1_norm, pts2_norm):
            x, y = pt1
            x_p, y_p = pt2
            A.append([-x, -y, -1, 0, 0, 0, x * x_p, y * x_p, x_p])
            A.append([0, 0, 0, -x, -y, -1, x * y_p, y * y_p, y_p])
        A = np.array(A)

        try:
            _, _, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            logger.warning("SVD failed. Returning None for homography.")
            return None

        H_norm = Vt[-1].reshape(3, 3)
        H = np.linalg.inv(T2) @ H_norm @ T1
        return H / H[2, 2]

    def compute_homography_using_ransac(self, pts1, pts2):
        max_iters, threshold = 500, 3.0
        optimal_H, max_inliers, optimal_inliers = None, 0, []

        if len(pts1) < 4:
            return None

        for _ in range(max_iters):
            indices = np.random.choice(len(pts1), 4, replace=False)
            pts1_sample, pts2_sample = pts1[indices], pts2[indices]

            candidate_H = self.direct_linear_transform(pts1_sample, pts2_sample)
            if candidate_H is None:
                continue

            pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
            proj_pts2 = (candidate_H @ pts1_h.T).T
            proj_pts2[:, 2][proj_pts2[:, 2] == 0] = 1e-10
            proj_pts2 = proj_pts2[:, :2] / proj_pts2[:, 2, np.newaxis]

            error = np.linalg.norm(pts2 - proj_pts2, axis=1)
            inliers = np.where(error < threshold)[0]

            if len(inliers) > max_inliers:
                max_inliers, optimal_H, optimal_inliers = len(inliers), candidate_H, inliers

            if len(inliers) > 0.7 * len(pts1):
                break

        if optimal_H is not None and len(optimal_inliers) >= 10:
            optimal_H = self.direct_linear_transform(pts1[optimal_inliers], pts2[optimal_inliers])
        else:
            logger.warning("Not enough inliers after RANSAC.")
            return None

        return optimal_H

    def apply_homography_to_individual_points(self, H, points):
        # Convert to homogeneous coordinates and apply transformation
        homogeneous_pts = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_pts = (H @ homogeneous_pts.T).T
        transformed_pts[:, 2][transformed_pts[:, 2] == 0] = 1e-10
        transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2, np.newaxis]
        return transformed_pts

    def warp_image(self, img1, img2, H, shape):
        h_out, w_out = shape
        grid_x, grid_y = np.meshgrid(np.arange(w_out), np.arange(h_out))
        coords = np.stack([grid_x.ravel(), grid_y.ravel(), np.ones_like(grid_x).ravel()], axis=-1)

        # Apply inverse homography to get source coordinates
        H_inv = np.linalg.inv(H)
        coords_transformed = coords @ H_inv.T
        coords_transformed[:, 2][coords_transformed[:, 2] == 0] = 1e-10
        coords_transformed /= coords_transformed[:, 2, np.newaxis]

        x_src, y_src = coords_transformed[:, 0], coords_transformed[:, 1]
        valid = (x_src >= 0) & (x_src < img2.shape[1] - 1) & (y_src >= 0) & (y_src < img2.shape[0] - 1)

        x_src, y_src = x_src[valid], y_src[valid]
        x0, y0 = np.floor(x_src).astype(int), np.floor(y_src).astype(int)
        x1, y1 = x0 + 1, y0 + 1

        wx, wy = x_src - x0, y_src - y0
        img_flat = img2.reshape(-1, img2.shape[2])

        Ia = img_flat[y0 * img2.shape[1] + x0]
        Ib = img_flat[y0 * img2.shape[1] + x1]
        Ic = img_flat[y1 * img2.shape[1] + x0]
        Id = img_flat[y1 * img2.shape[1] + x1]

        warped_img = (Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy) +
                      Ic * (1 - wx) * wy + Id * wx * wy)

        output_img = np.zeros((h_out * w_out, img2.shape[2]), dtype=img2.dtype)
        output_img[valid] = warped_img
        return output_img.reshape(h_out, w_out, img2.shape[2])

    def inverse_warp(self, img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        transformed_corners = self.apply_homography_to_individual_points(H, corners)
        all_corners = np.vstack((transformed_corners, [[0, 0], [w1, 0], [w1, h1], [0, h1]]))
        min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)

        # Define translation matrix and adjust homography
        translation_mat = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        H_translated = translation_mat @ H

        out_shape = (max_y - min_y, max_x - min_x)
        warped_img2 = self.warp_image(img1, img2, H_translated, out_shape)

        composite_img = np.zeros((out_shape[0], out_shape[1], 3), dtype=img1.dtype)
        composite_img[-min_y:h1 - min_y, -min_x:w1 - min_x] = img1

        mask1 = (composite_img > 0).astype(np.float32)
        mask2 = (warped_img2 > 0).astype(np.float32)

        blend_mask = mask1 + mask2
        safe_blend = np.where(blend_mask == 0, 1, blend_mask)
        composite_img = (composite_img * mask1 + warped_img2 * mask2) / safe_blend
        return np.nan_to_num(composite_img).astype(np.uint8)
