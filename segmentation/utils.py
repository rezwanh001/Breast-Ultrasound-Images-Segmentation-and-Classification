#-*- coding: utf-8 -*-
"""
@author: Md. Rezwanul Haque
"""
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import os

def num_index(image) :
    """
    * Helper function: To get the index for real image and mask.

        0. Benign Image Numner: 109
        1. Malignant Image Number: 123

        -------

        Image extention Format:
        - Image : 1 Benign Image.bmp
        - Lesion: 1 Benign Lesion.bmp
        - Mask  : 1 Benign Mask.tif
    """
    num_split = image.split(" ")[0]
    image_type = image.split(" ")[1]
    image_mask_split = (image.split(" ")[2]).split(".")[0]
    
    return int(num_split), image_type, image_mask_split