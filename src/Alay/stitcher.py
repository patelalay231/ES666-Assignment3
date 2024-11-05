import cv2
import numpy as np
import glob
import logging

# Configure logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PanoramaLogger")

class PanoramaStitcher:
    def __init__(self):
        self.sift_detector = cv2.SIFT_create()
        flann_params = dict(algorithm=1, trees=5)
        search_criteria = dict(checks=50)
        self.feature_matcher = cv2.FlannBasedMatcher(flann_params, search_criteria)

    def stitch_images_from_directory(self, directory_path):
        image_files = glob.glob(f'{directory_path}/*.*')
        images = [cv2.imread(file) for file in image_files]
        
        # Start with the first image as the base
        panorama_image = images[0]
        homography_matrices = []

        for i in range(1, len(images)):
            keypoints_base, descriptors_base = self.sift_detector.detectAndCompute(panorama_image, None)
            keypoints_current, descriptors_current = self.sift_detector.detectAndCompute(images[i], None)

            if descriptors_base is None or descriptors_current is None:
                logger.warning(f"Descriptors missing in image {i}. Skipping.")
                continue

            matches = self.feature_matcher.knnMatch(descriptors_base, descriptors_current, k=2)
            # Apply ratio test to filter good matches
            valid_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(valid_matches) < 6:
                logger.warning(f"Insufficient matches between image {i} and image {i - 1}. Skipping.")
                continue

            pts_base = np.float32([keypoints_base[m.queryIdx].pt for m in valid_matches]).reshape(-1, 2)
            pts_current = np.float32([keypoints_current[m.trainIdx].pt for m in valid_matches]).reshape(-1, 2)

            homography = self.estimate_homography_ransac(pts_current, pts_base)
            if homography is None:
                logger.warning(f"Could not compute homography for images {i} and {i - 1}. Skipping.")
                continue

            homography_matrices.append(homography)
            panorama_image = self.apply_inverse_warp(panorama_image, images[i], homography)

        return panorama_image, homography_matrices

    def normalize_points(self, points):
        mean, std_dev = np.mean(points, axis=0), np.std(points, axis=0)
        std_dev[std_dev < 1e-8] = 1e-8
        scale = np.sqrt(2) / std_dev
        transformation_matrix = np.array([[scale[0], 0, -scale[0]*mean[0]],
                                          [0, scale[1], -scale[1]*mean[1]],
                                          [0, 0, 1]])
        homogeneous_pts = np.hstack((points, np.ones((points.shape[0], 1))))
        normalized_pts = (transformation_matrix @ homogeneous_pts.T).T
        return normalized_pts[:, :2], transformation_matrix

    def direct_linear_transform(self, src_pts, dst_pts):
        src_pts_norm, T_src = self.normalize_points(src_pts)
        dst_pts_norm, T_dst = self.normalize_points(dst_pts)
        A = []
        for src, dst in zip(src_pts_norm, dst_pts_norm):
            x, y = src
            x_prime, y_prime = dst
            A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        A = np.array(A)
        try:
            _, _, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            logger.warning("SVD computation failed.")
            return None
        homography_norm = Vt[-1].reshape(3, 3)
        homography = np.linalg.inv(T_dst) @ homography_norm @ T_src
        return homography / homography[2, 2]

    def estimate_homography_ransac(self, src_pts, dst_pts):
        max_iter, inlier_threshold = 500, 3.0
        best_homography, most_inliers, best_inlier_set = None, 0, []

        if len(src_pts) < 4:
            return None

        for _ in range(max_iter):
            selected_indices = np.random.choice(len(src_pts), 4, replace=False)
            src_sample = src_pts[selected_indices]
            dst_sample = dst_pts[selected_indices]

            candidate_homography = self.direct_linear_transform(src_sample, dst_sample)
            if candidate_homography is None:
                continue

            src_pts_homogeneous = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
            projected_dst_pts_homogeneous = (candidate_homography @ src_pts_homogeneous.T).T
            projected_dst_pts_homogeneous[:, 2][projected_dst_pts_homogeneous[:, 2] == 0] = 1e-10
            projected_dst_pts = projected_dst_pts_homogeneous[:, :2] / projected_dst_pts_homogeneous[:, 2, np.newaxis]

            errors = np.linalg.norm(dst_pts - projected_dst_pts, axis=1)
            inliers = np.where(errors < inlier_threshold)[0]

            if len(inliers) > most_inliers:
                most_inliers, best_homography, best_inlier_set = len(inliers), candidate_homography, inliers

            if len(inliers) > 0.7 * len(src_pts):
                break

        if best_homography is not None and len(best_inlier_set) >= 10:
            best_homography = self.direct_linear_transform(src_pts[best_inlier_set], dst_pts[best_inlier_set])
        else:
            logger.warning("Insufficient inliers after RANSAC.")
            return None

        return best_homography

    def warp_image(self, base_img, overlay_img, H, output_dim):
        h_out, w_out = output_dim
        x_indices, y_indices = np.meshgrid(np.arange(w_out), np.arange(h_out))
        ones = np.ones_like(x_indices)
        coordinates = np.stack([x_indices, y_indices, ones], axis=-1).reshape(-1, 3)

        H_inv = np.linalg.inv(H)
        transformed_coords = coordinates @ H_inv.T
        transformed_coords[:, 2][transformed_coords[:, 2] == 0] = 1e-10
        transformed_coords /= transformed_coords[:, 2, np.newaxis]

        x_src, y_src = transformed_coords[:, 0], transformed_coords[:, 1]
        valid = (x_src >= 0) & (x_src < overlay_img.shape[1] - 1) & (y_src >= 0) & (y_src < overlay_img.shape[0] - 1)

        x_src, y_src = x_src[valid], y_src[valid]
        x0, y0, x1, y1 = np.floor(x_src).astype(int), np.floor(y_src).astype(int), x0 + 1, y0 + 1
        wx, wy = x_src - x0, y_src - y0

        img_flat = overlay_img.reshape(-1, overlay_img.shape[2])
        indices = y0 * overlay_img.shape[1] + x0
        Ia, Ib, Ic, Id = img_flat[indices], img_flat[y0 * overlay_img.shape[1] + x1], img_flat[y1 * overlay_img.shape[1] + x0], img_flat[y1 * overlay_img.shape[1] + x1]

        warped_img = np.zeros((h_out * w_out, overlay_img.shape[2]), dtype=overlay_img.dtype)
        wa, wb, wc, wd = (1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy
        warped_img[valid] = (Ia * wa[:, np.newaxis] + Ib * wb[:, np.newaxis] + Ic * wc[:, np.newaxis] + Id * wd[:, np.newaxis])
        return warped_img.reshape(h_out, w_out, overlay_img.shape[2])

    def apply_inverse_warp(self, base_img, overlay_img, H):
        base_h, base_w = base_img.shape[:2]
        overlay_h, overlay_w = overlay_img.shape[:2]
        corners_overlay = np.array([[0, 0], [overlay_w, 0], [overlay_w, overlay_h], [0, overlay_h]])
        transformed_corners = self.apply_homography_to_individual_points(H, corners_overlay)
        all_corners = np.vstack((transformed_corners, [[0, 0], [base_w, 0], [base_w, base_h], [0, base_h]]))
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        translation_mat = np.array([[
