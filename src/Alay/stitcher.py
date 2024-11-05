import cv2
import numpy as np
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PanoramaLogger")

class PanaromaStitcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        flann_index_params = dict(algorithm=1, trees=5)
        flann_search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)

    def make_panaroma_for_images_in(self, path):
        image_paths = glob.glob(f'{path}/*.*')
        images = [cv2.imread(image_path) for image_path in image_paths]

        # Initialize with the first image as the base for stitching
        stitched_image = images[0]
        homography_matrices = []

        for i in range(1, len(images)):
            kp_base, des_base = self.sift.detectAndCompute(stitched_image, None)
            kp_current, des_current = self.sift.detectAndCompute(images[i], None)

            # Skip images if descriptors are missing
            if des_base is None or des_current is None:
                logger.warning(f"No descriptors in image {i}. Skipping.")
                continue

            # Find matches and apply Lowe's ratio test for filtering
            knn_matches = self.matcher.knnMatch(des_base, des_current, k=2)
            good_matches = [m for m, n in knn_matches if m.distance < 0.7 * n.distance]

            # Ensure sufficient matches are found for reliable homography
            if len(good_matches) < 6:
                logger.warning(f"Insufficient good matches between image {i} and image {i-1}. Skipping.")
                continue

            # Extract matched keypoints
            pts_base = np.float32([kp_base[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            pts_current = np.float32([kp_current[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            # Compute homography matrix using RANSAC
            homography = self.compute_homography_using_ransac(pts_current, pts_base)
            if homography is None:
                logger.warning(f"Failed to compute homography between image {i} and image {i-1}. Skipping.")
                continue

            homography_matrices.append(homography)
            stitched_image = self.inverse_warp(stitched_image, images[i], homography)

        return stitched_image, homography_matrices

    def normalize_points(self, points):
        mean, std_dev = np.mean(points, axis=0), np.std(points, axis=0)
        std_dev[std_dev < 1e-8] = 1e-8  # Prevent division by zero
        scale = np.sqrt(2) / std_dev
        normalization_matrix = np.array([[scale[0], 0, -scale[0] * mean[0]],
                                         [0, scale[1], -scale[1] * mean[1]],
                                         [0, 0, 1]])
        homogeneous_pts = np.hstack((points, np.ones((points.shape[0], 1))))
        normalized_pts = (normalization_matrix @ homogeneous_pts.T).T
        return normalized_pts[:, :2], normalization_matrix

    def direct_linear_transform(self, pts1, pts2):
        pts1_normalized, T1 = self.normalize_points(pts1)
        pts2_normalized, T2 = self.normalize_points(pts2)
        A = []
        for pt1, pt2 in zip(pts1_normalized, pts2_normalized):
            x, y = pt1
            x_prime, y_prime = pt2
            A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        A = np.array(A)
        
        try:
            _, _, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            logger.warning("SVD computation failed.")
            return None

        homography_norm = Vt[-1].reshape(3, 3)
        homography = np.linalg.inv(T2) @ homography_norm @ T1
        return homography / homography[2, 2]

    def compute_homography_using_ransac(self, pts1, pts2):
        max_iterations, inlier_threshold = 500, 3.0
        best_homography, max_inliers, best_inliers = None, 0, []

        if len(pts1) < 4:
            return None

        for _ in range(max_iterations):
            sample_indices = np.random.choice(len(pts1), 4, replace=False)
            sample_pts1, sample_pts2 = pts1[sample_indices], pts2[sample_indices]
            
            candidate_homography = self.direct_linear_transform(sample_pts1, sample_pts2)
            if candidate_homography is None:
                continue

            # Project points to test homography inliers
            pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
            projected_pts2 = (candidate_homography @ pts1_homogeneous.T).T
            projected_pts2[:, 2][projected_pts2[:, 2] == 0] = 1e-10
            projected_pts2 = projected_pts2[:, :2] / projected_pts2[:, 2, np.newaxis]

            errors = np.linalg.norm(pts2 - projected_pts2, axis=1)
            inliers = np.where(errors < inlier_threshold)[0]

            if len(inliers) > max_inliers:
                max_inliers, best_homography, best_inliers = len(inliers), candidate_homography, inliers

            if len(inliers) > 0.7 * len(pts1):
                break

        if best_homography is not None and len(best_inliers) >= 10:
            best_homography = self.direct_linear_transform(pts1[best_inliers], pts2[best_inliers])
        else:
            logger.warning("Insufficient inliers found with RANSAC.")
            return None

        return best_homography

    def apply_homography_to_individual_points(self, H, points):
        homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points = (H @ homogeneous_points.T).T
        transformed_points[:, 2][transformed_points[:, 2] == 0] = 1e-10
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2, np.newaxis]
        return transformed_points

    def warp_image(self, base_img, overlay_img, H, output_shape):
        h_out, w_out = output_shape
        x_indices, y_indices = np.meshgrid(np.arange(w_out), np.arange(h_out))
        ones = np.ones_like(x_indices)
        coordinates = np.stack([x_indices, y_indices, ones], axis=-1).reshape(-1, 3)

        H_inv = np.linalg.inv(H)
        transformed_coords = coordinates @ H_inv.T
        transformed_coords[:, 2][transformed_coords[:, 2] == 0] = 1e-10
        transformed_coords /= transformed_coords[:, 2, np.newaxis]

        x_src, y_src = transformed_coords[:, 0], transformed_coords[:, 1]
        valid_indices = (x_src >= 0) & (x_src < overlay_img.shape[1] - 1) & \
                        (y_src >= 0) & (y_src < overlay_img.shape[0] - 1)

        x_src, y_src = x_src[valid_indices], y_src[valid_indices]
        x0, y0 = np.floor(x_src).astype(int), np.floor(y_src).astype(int)
        x1, y1 = x0 + 1, y0 + 1

        wx, wy = x_src - x0, y_src - y0
        img_flat = overlay_img.reshape(-1, overlay_img.shape[2])
        Ia, Ib, Ic, Id = img_flat[y0 * overlay_img.shape[1] + x0], \
                         img_flat[y0 * overlay_img.shape[1] + x1], \
                         img_flat[y1 * overlay_img.shape[1] + x0], \
                         img_flat[y1 * overlay_img.shape[1] + x1]

        warped_pixels = (Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy) +
                         Ic * (1 - wx) * wy + Id * wx * wy)
        warped_img = np.zeros((h_out * w_out, overlay_img.shape[2]), dtype=overlay_img.dtype)
        warped_img[valid_indices] = warped_pixels
        return warped_img.reshape(h_out, w_out, overlay_img.shape[2])

    def inverse_warp(self, img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        transformed_corners = self.apply_homography_to_individual_points(H, corners_img2)
        all_corners = np.vstack((transformed_corners, [[0, 0], [w1, 0], [w1, h1], [0, h1]]))
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        H_translated = translation @ H

        output_shape = (y_max - y_min, x_max - x_min)
        warped_img2 = self.warp_image(img1, img2, H_translated, output_shape)

        stitched_img = np.zeros((output_shape[0], output_shape[1], 3), dtype=img1.dtype)
        stitched_img[-y_min:-y_min + h1, -x_min:-x_min + w1] = img1

        mask1, mask2 = (stitched_img > 0).astype(np.float32), (warped_img2 > 0).astype(np.float32)
        combined_mask = mask1 + mask2
        safe_combined_mask = np.where(combined_mask == 0, 1, combined_mask)
        stitched_img = (stitched_img * mask1 + warped_img2 * mask2) / safe_combined_mask
        return np.nan_to_num(stitched_img).astype(np.uint8)
