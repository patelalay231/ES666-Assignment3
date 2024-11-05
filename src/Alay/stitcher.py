import pdb
import glob
import cv2
import os
import numpy as np
from src.JohnDoe import some_function
from src.JohnDoe.some_folder import folder_func
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PanaromaStitcher():
    def __init__(self):
        """Initialize the PanoramaStitcher with SIFT detector and matcher"""
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.min_matches = 10
        self.ratio = 0.7

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        
        if len(all_images) < 2:
            raise ValueError("At least 2 images are required for stitching")
        
        # Read all images
        images = [cv2.imread(im) for im in all_images]
        if any(img is None for img in images):
            raise ValueError("Failed to read one or more images")
            
        # Initialize with first image
        stitched_image = images[0]
        homography_matrix_list = []
        
        # Process each subsequent image
        for i in range(1, len(images)):
            logger.info(f"Processing image pair {i}/{len(images)-1}")
            
            # Get keypoints and descriptors
            kp1, des1 = self.sift.detectAndCompute(stitched_image, None)
            kp2, des2 = self.sift.detectAndCompute(images[i], None)
            
            if des1 is None or des2 is None:
                logger.warning(f"No features found in image pair {i}. Skipping.")
                continue
                
            # Match features
            matches = self.match_features(des1, des2)
            
            if len(matches) < self.min_matches:
                logger.warning(f"Not enough matches in image pair {i}. Skipping.")
                continue
                
            # Get matching points
            pts1, pts2 = self.get_matching_points(kp1, kp2, matches)
            
            # Compute homography
            H = self.compute_homography(pts1, pts2)
            if H is None:
                logger.warning(f"Failed to compute homography for image pair {i}. Skipping.")
                continue
                
            homography_matrix_list.append(H)
            stitched_image = self.warp_images(stitched_image, images[i], H)
            
        return stitched_image, homography_matrix_list

    def match_features(self, des1, des2):
        """Match features between two images using ratio test"""
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio * n.distance:
                good_matches.append(m)
        return good_matches

    def get_matching_points(self, kp1, kp2, matches):
        """Extract matching points from keypoints and matches"""
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        return pts1, pts2

    def compute_homography(self, pts1, pts2):
        """Compute homography matrix using RANSAC"""
        if len(pts1) < 4:
            return None
            
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None:
            return None
            
        # Check if homography is valid
        if not self.is_valid_homography(H):
            return None
            
        return H

    def is_valid_homography(self, H):
        """Check if homography matrix is valid"""
        # Check if matrix is not singular
        if np.linalg.det(H) == 0:
            return False
            
        # Check if transformation is not too extreme
        if np.linalg.norm(H) > 4.0:
            return False
            
        return True

    def warp_images(self, img1, img2, H):
        """Warp and blend images using homography"""
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Calculate corners of warped image
        corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners_transformed = cv2.perspectiveTransform(corners, H)
        corners_transformed = np.concatenate(
            (corners_transformed, np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2))
        )
        
        # Calculate dimensions of output image
        [x_min, y_min] = np.int32(corners_transformed.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(corners_transformed.max(axis=0).ravel() + 0.5)
        
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
        result = self.blend_images(warped_img1, warped_img2)
        
        return result

    def blend_images(self, img1, img2):
        """Blend two images with overlapping regions"""
        # Create masks for non-zero pixels
        mask1 = (img1 != 0).any(axis=2)
        mask2 = (img2 != 0).any(axis=2)
        overlap = mask1 & mask2
        
        # Initialize result
        result = np.zeros_like(img1)
        
        # Copy non-overlapping pixels
        result[mask1 & ~overlap] = img1[mask1 & ~overlap]
        result[mask2 & ~overlap] = img2[mask2 & ~overlap]
        
        # Blend overlapping pixels
        result[overlap] = (img1[overlap].astype(np.float32) + 
                         img2[overlap].astype(np.float32)) / 2
        
        return result.astype(np.uint8)

    def say_hi(self):
        """Print greeting message"""
        print('Hii From Jane Doe..')
    
    def do_something(self):
        """Placeholder method"""
        return None
    
    def do_something_more(self):
        """Placeholder method"""
        return None