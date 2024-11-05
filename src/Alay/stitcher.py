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
        img_paths = glob.glob('{}/*.*'.format(path))
        image_sequence = [cv2.imread(im_path) for im_path in img_paths]
        result_img = image_sequence[0]
        transform_matrices = []

        for idx in range(1, len(image_sequence)):
            keypoints1, desc1 = self.sift.detectAndCompute(result_img, None)
            keypoints2, desc2 = self.sift.detectAndCompute(image_sequence[idx], None)
            
            if desc1 is None or desc2 is None:
                logger.warning(f"No descriptors found in image {idx}. Skipping this pair.")
                continue
                
            matched_pairs = self.matcher.knnMatch(desc1, desc2, k=2)
            good_points = []
            for match1, match2 in matched_pairs:
                if match1.distance < 0.70 * match2.distance:
                    good_points.append(match1)

            if len(good_points) < 6:
                logger.warning(f"Not enough good matches between image {idx} and image {idx-1}. Skipping this pair.")
                continue
                
            points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_points]).reshape(-1, 2)
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_points]).reshape(-1, 2)

            transform_matrix = self.compute_homography_using_ransac(points2, points1)
            if transform_matrix is None:
                logger.warning(f"Failed to compute homography for image {idx} and image {idx-1}. Skipping this pair.")
                continue

            transform_matrices.append(transform_matrix)
            result_img = self.inverse_warp(result_img, image_sequence[idx], transform_matrix)

        return result_img, transform_matrices

    import numpy as np

    def normalize_points_with_matrix(self, pts):
        
        mean_x, mean_y = np.mean(pts, axis=0)
        std_x, std_y = np.std(pts, axis=0)
        
        
        if std_x < 1e-8:
            std_x = 1e-8
        if std_y < 1e-8:
            std_y = 1e-8
        
        
        scale_x = np.sqrt(2) / std_x
        scale_y = np.sqrt(2) / std_y
        
        
        T = np.array([
            [scale_x, 0, -scale_x * mean_x],
            [0, scale_y, -scale_y * mean_y],
            [0, 0, 1]
        ])
        
        
        pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
        
        
        normalized_pts_homogeneous = (T @ pts_homogeneous.T).T
        
        return normalized_pts_homogeneous[:, :2], T


    def direct_linear_transform(self, pts1, pts2):
        pts1_norm, T1 = self.normalize_points(pts1)
        pts2_norm, T2 = self.normalize_points(pts2)
        
        A = []
        for i in range(len(pts1_norm)):
            x, y = pts1_norm[i]
            x_prime, y_prime = pts2_norm[i]
            A.append([-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime])
            
        A = np.array(A)
        
        try:
            U, S, Vt = np.linalg.svd(A)
            H_norm = Vt[-1].reshape(3, 3)
            H = np.linalg.inv(T2) @ H_norm @ T1
            return H / H[2, 2]
        except np.linalg.LinAlgError:
            logger.warning("SVD did not converge. Returning None for homography.")
            return None

    def compute_homography_using_ransac(self, pts1, pts2):
        max_iterations = 500  
        threshold = 3.0
        best_H = None
        max_inliers = 0
        best_inliers = []

        if len(pts1) < 4:
            return None

        for iteration in range(max_iterations):
            idx = np.random.choice(len(pts1), 4, replace=False)
            p1_sample = pts1[idx]
            p2_sample = pts2[idx]

            H_candidate = self.direct_linear_transform(p1_sample, p2_sample)
            if H_candidate is None:
                continue

            pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
            projected_pts2_homogeneous = (H_candidate @ pts1_homogeneous.T).T
            
            projected_pts2_homogeneous[projected_pts2_homogeneous[:, 2] == 0, 2] = 1e-10
            projected_pts2 = projected_pts2_homogeneous[:, :2] / projected_pts2_homogeneous[:, 2, np.newaxis]
            
            errors = np.linalg.norm(pts2 - projected_pts2, axis=1)
            inliers = np.where(errors < threshold)[0]

            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_H = H_candidate
                best_inliers = inliers

            if len(inliers) > 0.7 * len(pts1):
                break

        if best_H is not None and len(best_inliers) >= 10:
            best_H = self.direct_linear_transform(pts1[best_inliers], pts2[best_inliers])
        else:
            logger.warning("Not enough inliers after ransac.")
            return None

        return best_H

    def apply_homography_to_individual_points(self, H, pts):
        pts_homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
        transformed_pts = (H @ pts_homogeneous.T).T
        transformed_pts[transformed_pts[:, 2] == 0, 2] = 1e-10
        transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2, np.newaxis]
        return transformed_pts

    def warp_image(self, img1, img2, H, output_shape):
        h_out, w_out = output_shape    
        xx, yy = np.meshgrid(np.arange(w_out), np.arange(h_out))
        ones = np.ones_like(xx)
        coords = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3)

        H_inv = np.linalg.inv(H)
        coords_transformed = coords @ H_inv.T
        coords_transformed[coords_transformed[:, 2] == 0, 2] = 1e-10
        coords_transformed /= coords_transformed[:, 2, np.newaxis]

        x_src = coords_transformed[:, 0]  
        y_src = coords_transformed[:, 1]

        valid_indices = (
            (x_src >= 0) & (x_src < img2.shape[1] - 1) &
            (y_src >= 0) & (y_src < img2.shape[0] - 1)
        )

        x_src = x_src[valid_indices]
        y_src = y_src[valid_indices]
        x0 = np.floor(x_src).astype(np.int32)
        y0 = np.floor(y_src).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        wx = x_src - x0        
        wy = y_src - y0

        img_flat = img2.reshape(-1, img2.shape[2])
        indices = y0 * img2.shape[1] + x0
        Ia = img_flat[indices]
        Ib = img_flat[y0 * img2.shape[1] + x1]
        Ic = img_flat[y1 * img2.shape[1] + x0]
        Id = img_flat[y1 * img2.shape[1] + x1]

        wa = (1 - wx) * (1 - wy)
        wb = wx * (1 - wy)
        wc = (1 - wx) * wy
        wd = wx * wy
        warped_pixels = (Ia * wa[:, np.newaxis] + Ib * wb[:, np.newaxis] +
                         Ic * wc[:, np.newaxis] + Id * wd[:, np.newaxis])

        warped_image = np.zeros((h_out * w_out, img2.shape[2]), dtype=img2.dtype)
        warped_image[valid_indices] = warped_pixels
        warped_image = warped_image.reshape(h_out, w_out, img2.shape[2])

        return warped_image

    def inverse_warp(self, img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        transformed_corners = self.apply_homography_to_individual_points(H, corners_img2)
        all_corners = np.vstack((transformed_corners, [[0, 0], [w1, 0], [w1, h1], [0, h1]]))
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        translation = np.array([[1, 0, -x_min],
                              [0, 1, -y_min],
                              [0, 0, 1]])
                              
        H_translated = translation @ H
        output_shape = (y_max - y_min, x_max - x_min)
        warped_img2 = self.warp_image(img1, img2, H_translated, output_shape)
        stitched_image = np.zeros((output_shape[0], output_shape[1], 3), dtype=img1.dtype)
        stitched_image[-y_min:-y_min + h1, -x_min:-x_min + w1] = img1

        mask1 = (stitched_image > 0).astype(np.float32)
        mask2 = (warped_img2 > 0).astype(np.float32)
        combined_mask = mask1 + mask2
        safe_combined_mask = np.where(combined_mask == 0, 1, combined_mask)
        stitched_image = (stitched_image * mask1 + warped_img2 * mask2) / safe_combined_mask
        stitched_image = np.nan_to_num(stitched_image).astype(np.uint8)

        return stitched_image