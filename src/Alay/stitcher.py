import cv2
import numpy as np
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PanaromaStitcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def make_panaroma_for_images_in(self, path):
        image_files = glob.glob(f'{path}/*.*')
        source_images = [cv2.imread(img_path) for img_path in image_files if cv2.imread(img_path) is not None]
        
        if not source_images:
            logger.warning("No valid images found in the directory.")
            return None, []

        final_image = source_images[0]
        transformation_matrices = []

        for current_idx in range(1, len(source_images)):
            current_image = source_images[current_idx]

            # Extract features
            features1, desc1 = self.sift.detectAndCompute(final_image, None)
            features2, desc2 = self.sift.detectAndCompute(current_image, None)

            if desc1 is None or desc2 is None:
                logger.warning(f"Feature detection failed for image {current_idx}")
                continue

            # Match features using KNN
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            filtered_matches = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]

            if len(filtered_matches) < 6:
                logger.warning(f"Insufficient matches for image {current_idx}")
                continue

            # Extract matching points
            source_points = np.float32([features1[m.queryIdx].pt for m in filtered_matches])
            target_points = np.float32([features2[m.trainIdx].pt for m in filtered_matches])

            # Calculate homography using RANSAC
            transform_matrix = self.compute_homography_using_ransac(target_points, source_points)
            if transform_matrix is None:
                logger.warning(f"Failed to compute transformation for image {current_idx}")
                continue

            transformation_matrices.append(transform_matrix)
            final_image = self.inverse_warp(final_image, current_image, transform_matrix)

        return final_image, transformation_matrices

    def compute_homography_using_ransac(self, pts1, pts2):
        iterations = 500
        distance_threshold = 3.0
        best_matrix = None
        max_inliers = 0

        if len(pts1) < 4:
            return None

        for _ in range(iterations):
            sample_idx = np.random.choice(len(pts1), 4, replace=False)
            sample_src = pts1[sample_idx]
            sample_dst = pts2[sample_idx]

            candidate_H = self.direct_linear_transform(sample_src, sample_dst)
            if candidate_H is None:
                continue

            homogeneous_pts = np.column_stack([pts1, np.ones(len(pts1))])
            projected_pts = (candidate_H @ homogeneous_pts.T).T

            projected_pts[projected_pts[:, 2] == 0, 2] = 1e-10
            projected_pts = projected_pts[:, :2] / projected_pts[:, 2, np.newaxis]

            distances = np.linalg.norm(pts2 - projected_pts, axis=1)
            inliers = distances < distance_threshold
            inlier_count = np.sum(inliers)

            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_matrix = candidate_H

            if inlier_count > 0.7 * len(pts1):
                break

        if best_matrix is not None and max_inliers >= 10:
            refined_H = self.direct_linear_transform(pts1[inliers], pts2[inliers])
            return refined_H
        
        logger.warning("RANSAC failed to find a good homography")
        return None

    def inverse_warp(self, img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        warped_corners = self.apply_homography_to_individual_points(H, corners)
        
        all_corners = np.vstack([warped_corners, [[0, 0], [w1, 0], [w1, h1], [0, h1]]])
        min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)

        translation = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])
        adjusted_H = translation @ H

        output_shape = (max_y - min_y, max_x - min_x)
        warped_img2 = self.warp_image(img1, img2, adjusted_H, output_shape)

        result = np.zeros_like(warped_img2)
        result[-min_y:-min_y + h1, -min_x:-min_x + w1] = img1

        mask1 = (result > 0).astype(np.float32)
        mask2 = (warped_img2 > 0).astype(np.float32)
        combined_mask = mask1 + mask2
        combined_mask[combined_mask == 0] = 1

        blended = (result * mask1 + warped_img2 * mask2) / combined_mask
        return np.nan_to_num(blended).astype(np.uint8)
