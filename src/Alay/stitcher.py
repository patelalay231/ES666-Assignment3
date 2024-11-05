import pdb
import glob
import cv2
import os

import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from matplotlib import style
from natsort import natsorted
from tqdm import tqdm
import random

class PanaromaStitcher():
    def __init__(self):
        pass

    def show_image_grid(self, images, M=1, N=1, title='Title', figsize=8):
      # Assuming 'images' is a numpy array of shape (num_images, height, width, channels)
      if M==1:
          row_size = figsize
          col_size = figsize//4
      elif N==1:
          row_size = figsize//4
          col_size = figsize
      else:
          row_size, col_size = figsize, figsize

      fig, axes = plt.subplots(M, N, figsize=(row_size, col_size))

      if len(images.shape) < 4:
          images = np.expand_dims(images.copy(), axis=0)

      for i in range(M):
          for j in range(N):
              if M==1 and N==1:
                  ax = axes
              elif M == 1 or N==1:
                  ax = axes[max(i, j)]
              else:
                  ax = axes[i, j]
              index = i * N + j
              if index < images.shape[0]:
                  ax.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
              ax.axis('off')
      plt.tight_layout()
      plt.show()

    def feature_detector(self, image1, image2, algo="blob", nfeatures=10000):
      img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
      img2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

      if algo == "blob":
        orb = cv2.ORB_create(nfeatures=nfeatures)
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        src = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,2)
      else:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(image1,None)
        kp2, des2 = sift.detectAndCompute(image2,None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        good = []
        for m in matches:
            if (m[0].distance < 0.5*m[1].distance):
                good.append(m)
        matches = np.asarray(good)

        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

      return src, dst

    def compute_homography(self, pointR, pointL):
      if len(pointL) < 4:
        raise ValueError("Not enough points")

      A = []
      for i in range(len(pointL)):
          x, y = pointR.copy()[i][0], pointR.copy()[i][1]
          x_prime, y_prime = pointL.copy()[i][0], pointL.copy()[i][1]

          A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
          A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])

      U, S, V = np.linalg.svd(np.array(A))
      H = V[-1].reshape(3, 3)
      H = H / H[2, 2]
      return H

    ## Refrence
    # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    ## H*pointL = pointR or H*x1 = x2
    def ransac(self, src_points, dst_points, point_selection = 4, iterations = 10000, threshold = 4.0):
        best_homography = None
        max_inliers_count = 0
        best_inliers = None

        for _ in range(iterations):
            indices = random.sample(range(len(src_points)), point_selection)
            src_sample = src_points[indices]
            dst_sample = dst_points[indices]

            H = self.compute_homography(src_sample, dst_sample)

            ## Check the score
            src_homogenous = np.hstack((src_points, np.ones((len(src_points), 1))))
            dst_projected = (H @ src_homogenous.T).T
            dst_projected /= dst_projected[:, 2].reshape(-1, 1)

            distances = np.sqrt(np.sum((dst_projected[:, :2] - dst_points) ** 2, axis=1))
            inliers = distances < threshold
            inliers_count = np.sum(inliers)

            if inliers_count > max_inliers_count:
                max_inliers_count = inliers_count
                best_homography = H

        return best_homography, max_inliers_count    

    ## Need to recheck this
    def warpPerspective(self, imageR, imageL, size_x, size_y, H):
      ## Going to transform imageR as per the imageL

      ## image2.shape == (height, width, channel)
      h, w = size_y, size_x
      output_image = np.zeros((size_y, size_x, 3), dtype=imageR.dtype)

      hinv = np.linalg.inv(H)

      for x in range(size_x):
          for y in range(size_y):
            co = np.array([x, y, 1], dtype=np.float32)  ### Somehow gets inverted
            tran_co = hinv @ co    ## 3x1 matrix
            tran_co /= tran_co[2]

            src_x, src_y = int(tran_co[0]), int(tran_co[1])

            if 0 <= (src_x) < (imageR.shape[1]) and 0 <= src_y < imageR.shape[0]:
                output_image[y, x] = imageR[src_y, src_x]

      return output_image

    def remove_Pad(self, image):
      rows = np.any(image != 0, axis=1)
      cols = np.any(image != 0, axis=0)

      r_min, r_max = np.where(rows)[0][[0, -1]]
      c_min, c_max = np.where(cols)[0][[0, -1]]

      cropped_image = image[r_min:r_max+1, c_min:c_max+1]

      return cropped_image

    def make_panaroma_for_images_in(self,path):
        print(path)
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        ## Take image
        gt_n_images = []
        gt_images = []
        for files in tqdm(natsorted(glob.glob(f"{path}/*"))):
            gt_images.append(cv2.imread(files, 1))

        gt_n_images.append(np.array(gt_images, dtype=object))
        
        homography_matrix_list = []
        ### Process
        img_num = 0
        ## Right to left
        tmp_img = gt_n_images[img_num][-1]
        tmp_img = cv2.resize(tmp_img.astype(np.uint8), (640, 480))
        
        for i in range(len(gt_n_images[img_num])-2, -1, -1):
          ## Image 1 should be the right most one.
          imageR = tmp_img.astype(np.uint8)
          imageL = gt_n_images[img_num][i].astype(np.uint8)
          imageL = cv2.resize(imageL, (640, 480))

          pointL, pointR  = self.feature_detector(imageL, imageR)

          break_point = 0
          while True:
            H, count = self.ransac(pointR.copy(), pointL.copy())
            
            if count >= 200 or break_point > 10:
              break
            break_point += 1

          print(count)
          if count < 200:
            break

          homography_matrix_list.append(H)

          output_image = self.warpPerspective(imageR, imageL, (imageR.shape[1] + imageL.shape[1]), 100 + np.max((imageR.shape[0], imageL.shape[0])), H)
          
          output_image_new = output_image.copy()
          output_image_new[:imageL.shape[0], :imageL.shape[1]] = imageL
          tmp_img = self.remove_Pad(output_image_new)

        return tmp_img, homography_matrix_list 

        self.say_hi()


        # Collect all homographies calculated for pair of images and return
        homography_matrix_list =[]
        # Return Final panaroma
        stitched_image = cv2.imread(all_images[0])
        #####
        
        return stitched_image, homography_matrix_list 

    def say_hi(self):
        raise NotImplementedError('I am an Error. Fix Me Please!')