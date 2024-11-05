import pdb
import glob
import cv2
import os

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        self.say_hi()


        # Collect all homographies calculated for pair of images and return
        homography_matrix_list =[]
        # Return Final panaroma
        stitched_image = cv2.imread(all_images[0])
        #####
        
        return stitched_image, homography_matrix_list 

    def say_hi(self):
        raise NotImplementedError('I am an Error. Fix Me Please!')
