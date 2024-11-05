import cv2
import numpy as np
import glob
import logging
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PanoramaStitcher:
    def __init__(self, ratio: float = 0.70, min_matches: int = 6):
        """
        Initialize PanoramaStitcher with parameters for feature matching
        
        Args:
            ratio: Lowe's ratio test threshold
            min_matches: Minimum number of good matches required
        """
        self.ratio = ratio
        self.min_matches = min_matches
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
    
    def compute_homography_using_ransac(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute homography matrix using RANSAC
        
        Args:
            src_pts: Source points (Nx2 array)
            dst_pts: Destination points (Nx2 array)
            
        Returns:
            Homography matrix or None if computation fails
        """
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None
            
        # RANSAC parameters
        max_iterations = 1000
        threshold = 5.0
        best_inliers = 0
        best_H = None
        
        for _ in range(max_iterations):
            # Randomly select 4 points
            idx = np.random.choice(len(src_pts), 4, replace=False)
            pts_src = src_pts[idx]
            pts_dst = dst_pts[idx]
            
            # Calculate homography using these points
            H = cv2.getPerspectiveTransform(
                pts_src.astype(np.float32).reshape(-1, 1, 2),
                pts_dst.astype(np.float32).reshape(-1, 1, 2)
            )
            
            # Transform all points using this homography
            src_pts_transformed = cv2.perspectiveTransform(
                src_pts.reshape(-1, 1, 2),
                H
            ).reshape(-1, 2)
            
            # Calculate distances and count inliers
            distances = np.linalg.norm(dst_pts - src_pts_transformed, axis=1)
            inliers = np.sum(distances < threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H
        
        # If we found a good homography with enough inliers
        if best_inliers >= self.min_matches:
            return best_H
        return None

    def inverse_warp(self, img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Warp img2 onto img1 using inverse warping
        
        Args:
            img1: First image (target)
            img2: Second image (to be warped)
            H: Homography matrix
            
        Returns:
            Stitched image
        """
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Find corners of second image after transform
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners2_transformed = cv2.perspectiveTransform(corners2, H)
        
        # Find dimensions of combined image
        all_corners = np.concatenate((
            np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
            corners2_transformed
        ))
        
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Translation matrix
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])
        
        # Warp images
        output_shape = (y_max - y_min, x_max - x_min)
        warped_img2 = cv2.warpPerspective(img2, translation @ H, output_shape)
        
        # Create translated version of img1
        warped_img1 = np.zeros_like(warped_img2)
        warped_img1[-y_min:h1-y_min, -x_min:w1-x_min] = img1
        
        # Blend images
        mask1 = (warped_img1 != 0).any(axis=2)
        mask2 = (warped_img2 != 0).any(axis=2)
        overlap = mask1 & mask2
        
        result = np.zeros_like(warped_img1)
        result[mask1] = warped_img1[mask1]
        result[mask2 & ~mask1] = warped_img2[mask2 & ~mask1]
        result[overlap] = (warped_img1[overlap].astype(np.float32) + 
                         warped_img2[overlap].astype(np.float32)) / 2
        
        return result.astype(np.uint8)

    def make_panorama_for_images_in(self, path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Create a panorama from all images in the specified directory
        
        Args:
            path: Directory path containing the images
            
        Returns:
            Tuple of (stitched_image, list_of_homography_matrices)
        """
        # Get all image paths
        image_paths = glob.glob('{}/*.*'.format(path))
        if not image_paths:
            raise ValueError(f"No images found in directory: {path}")
            
        # Read all images
        logger.info(f"Found {len(image_paths)} images in {path}")
        images = [cv2.imread(im_path) for im_path in image_paths]
        
        # Start with the first image
        stitched_image = images[0]
        homography_matrix_list = []
        
        # Iterate through remaining images
        for i in range(1, len(images)):
            logger.info(f"Processing image {i+1} of {len(images)}")
            
            # Detect features
            kp1, des1 = self.sift.detectAndCompute(stitched_image, None)
            kp2, des2 = self.sift.detectAndCompute(images[i], None)
            
            # Check if features were found
            if des1 is None or des2 is None:
                logger.warning(f"No descriptors found in image {i}. Skipping this pair.")
                continue
                
            # Match features
            knn_matches = self.matcher.knnMatch(des1, des2, k=2)
            good_matches = []
            
            # Apply Lowe's ratio test
            for m, n in knn_matches:
                if m.distance < self.ratio * n.distance:
                    good_matches.append(m)
                    
            # Check if enough matches were found
            if len(good_matches) < self.min_matches:
                logger.warning(f"Not enough good matches between image {i} and image {i-1}. "
                             f"Found {len(good_matches)}, needed {self.min_matches}. "
                             "Skipping this pair.")
                continue
                
            # Extract matching points
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
            
            # Compute homography
            H = self.compute_homography_using_ransac(pts2, pts1)
            
            if H is None:
                logger.warning(f"Failed to compute homography for image {i} and image {i-1}. "
                             "Skipping this pair.")
                continue
                
            # Store homography and stitch images
            homography_matrix_list.append(H)
            stitched_image = self.inverse_warp(stitched_image, images[i], H)
            
            logger.info(f"Successfully stitched image {i+1}")
        
        return stitched_image, homography_matrix_list