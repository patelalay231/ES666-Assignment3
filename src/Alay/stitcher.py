import cv2
import numpy as np
from typing import List, Tuple, Optional

class PanoramaStitcher:
    def __init__(self, ratio: float = 0.75, min_matches: int = 10):
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
    
    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
        """
        Detect and match features between two images using SIFT
        """
        # Detect SIFT features
        kp1, desc1 = self.sift.detectAndCompute(img1, None)
        kp2, desc2 = self.sift.detectAndCompute(img2, None)
        
        # Match features using k-nearest neighbors
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio * n.distance:
                good_matches.append(m)
        
        return kp1, kp2, good_matches
    
    def find_homography(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], 
                       matches: List[cv2.DMatch]) -> Optional[np.ndarray]:
        """
        Calculate homography matrix using RANSAC
        """
        if len(matches) < self.min_matches:
            return None
            
        # Extract matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Implement RANSAC
        best_H = None
        best_inliers = 0
        iterations = 1000
        threshold = 5.0  # Distance threshold for inlier detection
        
        for _ in range(iterations):
            # Randomly select 4 point pairs
            if len(matches) < 4:
                return None
                
            idx = np.random.choice(len(matches), 4, replace=False)
            pts1 = src_pts[idx]
            pts2 = dst_pts[idx]
            
            # Calculate homography using DLT (Direct Linear Transform)
            H = self.compute_homography_dlt(pts1, pts2)
            if H is None:
                continue
                
            # Count inliers
            inliers = 0
            for i in range(len(matches)):
                # Transform point
                pt1 = np.float32([src_pts[i][0][0], src_pts[i][0][1], 1.0])
                transformed_pt = np.dot(H, pt1)
                transformed_pt = transformed_pt / transformed_pt[2]
                
                # Calculate distance
                actual_pt = np.float32([dst_pts[i][0][0], dst_pts[i][0][1]])
                estimated_pt = np.float32([transformed_pt[0], transformed_pt[1]])
                distance = np.linalg.norm(actual_pt - estimated_pt)
                
                if distance < threshold:
                    inliers += 1
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H
        
        return best_H
    
    def compute_homography_dlt(self, pts1: np.ndarray, pts2: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute homography matrix using Direct Linear Transform
        """
        if pts1.shape[0] < 4 or pts2.shape[0] < 4:
            return None
            
        A = []
        for i in range(pts1.shape[0]):
            x, y = pts1[i][0]
            u, v = pts2[i][0]
            A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
            A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        
        A = np.array(A)
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        
        # Normalize
        H = H / H[2, 2]
        return H
    
    def warp_images(self, img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Warp and blend images using the computed homography
        """
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Find corners of first image after transform
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        
        # Transform corners of img1
        corners1_transformed = cv2.perspectiveTransform(corners1, H)
        corners = np.concatenate((corners1_transformed, corners2), axis=0)
        
        # Find dimensions of combined image
        [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)
        
        # Translation matrix to shift to positive coordinates
        translation = np.array([[1, 0, -x_min],
                              [0, 1, -y_min],
                              [0, 0, 1]])
        
        H_translated = translation @ H
        
        # Warp images
        output_shape = (y_max - y_min, x_max - x_min)
        warped1 = cv2.warpPerspective(img1, H_translated, (output_shape[1], output_shape[0]))
        
        # Create translated version of img2
        warped2 = np.zeros_like(warped1)
        warped2[-y_min:h2-y_min, -x_min:w2-x_min] = img2
        
        # Blend images
        result = self.blend_images(warped1, warped2)
        
        return result
    
    def blend_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Blend two images using a simple average where they overlap
        """
        # Create masks for non-zero pixels
        mask1 = (img1 != 0).astype(np.float32)
        mask2 = (img2 != 0).astype(np.float32)
        
        # Calculate overlap mask
        overlap = mask1 * mask2
        
        # Initialize result array
        result = np.zeros_like(img1, dtype=np.float32)
        
        # Add images where there's no overlap
        result += img1 * (mask1 * (1 - overlap))
        result += img2 * (mask2 * (1 - overlap))
        
        # Average in overlap regions
        overlap_sum = overlap.sum(axis=2, keepdims=True)
        overlap_sum[overlap_sum == 0] = 1  # Avoid division by zero
        result += (img1 * overlap + img2 * overlap) / 2
        
        return result.astype(np.uint8)
    
    def stitch(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """
        Main method to stitch two images together
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Stitched panorama image or None if stitching fails
        """
        # Detect and match features
        kp1, kp2, matches = self.detect_and_match_features(img1, img2)
        
        if len(matches) < self.min_matches:
            return None
            
        # Find homography
        H = self.find_homography(kp1, kp2, matches)
        
        if H is None:
            return None
            
        # Warp and blend images
        result = self.warp_images(img1, img2, H)
        
        return result