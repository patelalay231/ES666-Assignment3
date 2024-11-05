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
        image_files = glob.glob('{}/*.*'.format(path))
        source_images = [cv2.imread(img_path) for img_path in image_files]
        final_image = source_images[0]
        transformation_matrices = []

        for current_idx in range(1, len(source_images)):
            current_image = source_images[current_idx]
            
            # Extract features
            features1, desc1 = self.sift.detectAndCompute(final_image, None)
            features2, desc2 = self.sift.detectAndCompute(current_image, None)
            
            if not all([desc1 is not None, desc2 is not None]):
                logger.warning(f"Feature detection failed for image {current_idx}")
                continue
                
            # Match features using KNN
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            filtered_matches = []
            for primary, secondary in raw_matches:
                if primary.distance < 0.70 * secondary.distance:
                    filtered_matches.append(primary)
                    
            if len(filtered_matches) < 6:
                logger.warning(f"Insufficient matches for image {current_idx}")
                continue
                
            # Extract matching points
            source_points = np.float32([features1[match.queryIdx].pt for match in filtered_matches])
            target_points = np.float32([features2[match.trainIdx].pt for match in filtered_matches])
            
            # Calculate homography
            transform_matrix = self.compute_homography_using_ransac(target_points, source_points)
            if transform_matrix is None:
                logger.warning(f"Failed to compute transformation for image {current_idx}")
                continue
                
            transformation_matrices.append(transform_matrix)
            final_image = self.inverse_warp(final_image, current_image, transform_matrix)

        return final_image, transformation_matrices

    def normalize_points(self, points):
        centroid = np.mean(points, axis=0)
        std_dev = np.std(points, axis=0)
        std_dev[std_dev < 1e-8] = 1e-8
        
        scaling = np.sqrt(2) / std_dev
        transform = np.array([
            [scaling[0], 0, -scaling[0] * centroid[0]],
            [0, scaling[1], -scaling[1] * centroid[1]],
            [0, 0, 1]
        ])
        
        homogeneous_coords = np.column_stack([points, np.ones(len(points))])
        normalized_points = (transform @ homogeneous_coords.T).T
        
        return normalized_points[:, :2], transform

    def direct_linear_transform(self, src_pts, dst_pts):
        if len(src_pts) < 4:
            return None
            
        src_normalized, T1 = self.normalize_points(src_pts)
        dst_normalized, T2 = self.normalize_points(dst_pts)
        
        equations = []
        for i in range(len(src_normalized)):
            x, y = src_normalized[i]
            u, v = dst_normalized[i]
            equations.extend([
                [-x, -y, -1, 0, 0, 0, x*u, y*u, u],
                [0, 0, 0, -x, -y, -1, x*v, y*v, v]
            ])
            
        equation_matrix = np.array(equations)
        
        try:
            _, _, Vt = np.linalg.svd(equation_matrix)
            H_normalized = Vt[-1].reshape(3, 3)
            H = np.linalg.inv(T2) @ H_normalized @ T1
            return H / H[2, 2]
        except np.linalg.LinAlgError:
            logger.warning("SVD computation failed")
            return None

    def compute_homography_using_ransac(self, pts1, pts2):
        iterations = 500
        distance_threshold = 3.0
        best_matrix = None
        max_inliers = 0
        best_inlier_mask = None

        if len(pts1) < 4:
            return None

        for _ in range(iterations):
            # Random sample selection
            sample_idx = np.random.choice(len(pts1), 4, replace=False)
            sample_src = pts1[sample_idx]
            sample_dst = pts2[sample_idx]

            # Compute candidate homography
            candidate_H = self.direct_linear_transform(sample_src, sample_dst)
            if candidate_H is None:
                continue

            # Project points and calculate errors
            homogeneous_pts = np.column_stack([pts1, np.ones(len(pts1))])
            projected_pts = (candidate_H @ homogeneous_pts.T).T
            
            # Handle division by zero
            projected_pts[projected_pts[:, 2] == 0, 2] = 1e-10
            projected_pts = projected_pts[:, :2] / projected_pts[:, 2, np.newaxis]
            
            # Calculate distances and find inliers
            distances = np.linalg.norm(pts2 - projected_pts, axis=1)
            inliers = distances < distance_threshold
            inlier_count = np.sum(inliers)

            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_matrix = candidate_H
                best_inlier_mask = inliers

            if inlier_count > 0.7 * len(pts1):
                break

        if best_matrix is not None and np.sum(best_inlier_mask) >= 10:
            # Refine homography using all inliers
            refined_H = self.direct_linear_transform(
                pts1[best_inlier_mask],
                pts2[best_inlier_mask]
            )
            return refined_H
        
        logger.warning("RANSAC failed to find good homography")
        return None

    def apply_homography_to_individual_points(self, H, points):
        homogeneous_points = np.column_stack([points, np.ones(len(points))])
        transformed = (H @ homogeneous_points.T).T
        transformed[transformed[:, 2] == 0, 2] = 1e-10
        return transformed[:, :2] / transformed[:, 2, np.newaxis]

    def warp_image(self, img1, img2, H, output_shape):
        height, width = output_shape
        y_coords, x_coords = np.meshgrid(range(height), range(width), indexing='ij')
        coords = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
        coords_reshaped = coords.reshape(-1, 3)

        # Transform coordinates
        H_inv = np.linalg.inv(H)
        transformed = coords_reshaped @ H_inv.T
        transformed[transformed[:, 2] == 0, 2] = 1e-10
        transformed /= transformed[:, 2, np.newaxis]

        x_src = transformed[:, 0]
        y_src = transformed[:, 1]

        # Find valid coordinates
        valid_coords = (
            (x_src >= 0) & (x_src < img2.shape[1] - 1) &
            (y_src >= 0) & (y_src < img2.shape[0] - 1)
        )

        x_src = x_src[valid_coords]
        y_src = y_src[valid_coords]

        # Bilinear interpolation
        x0 = np.floor(x_src).astype(np.int32)
        y0 = np.floor(y_src).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        # Calculate weights
        wx = x_src - x0
        wy = y_src - y0

        # Get pixel values
        img_flat = img2.reshape(-1, img2.shape[2])
        idx00 = y0 * img2.shape[1] + x0
        idx01 = y0 * img2.shape[1] + x1
        idx10 = y1 * img2.shape[1] + x0
        idx11 = y1 * img2.shape[1] + x1

        # Interpolate
        pixels = (
            img_flat[idx00] * ((1 - wx) * (1 - wy))[:, np.newaxis] +
            img_flat[idx01] * (wx * (1 - wy))[:, np.newaxis] +
            img_flat[idx10] * ((1 - wx) * wy)[:, np.newaxis] +
            img_flat[idx11] * (wx * wy)[:, np.newaxis]
        )

        # Create output image
        warped = np.zeros((height * width, img2.shape[2]), dtype=img2.dtype)
        warped[valid_coords] = pixels
        return warped.reshape(height, width, img2.shape[2])

    def inverse_warp(self, img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Find corners of warped image
        corners = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        warped_corners = self.apply_homography_to_individual_points(H, corners)
        
        # Calculate output dimensions
        all_corners = np.vstack([warped_corners, [[0, 0], [w1, 0], [w1, h1], [0, h1]]])
        min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)

        # Adjust transformation for new origin
        translation = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])
        adjusted_H = translation @ H

        # Create output image
        output_shape = (max_y - min_y, max_x - min_x)
        warped_img2 = self.warp_image(img1, img2, adjusted_H, output_shape)
        
        # Place first image in output space
        result = np.zeros_like(warped_img2)
        result[-min_y:-min_y + h1, -min_x:-min_x + w1] = img1

        # Blend images
        mask1 = (result > 0).astype(np.float32)
        mask2 = (warped_img2 > 0).astype(np.float32)
        combined_mask = mask1 + mask2
        combined_mask[combined_mask == 0] = 1

        blended = (result * mask1 + warped_img2 * mask2) / combined_mask
        return np.nan_to_num(blended).astype(np.uint8)