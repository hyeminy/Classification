import random
import math
import numbers

import cv2
import numpy as np
import torch

#https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks/blob/master/transforms/transforms.py

class ToCVImage:
    """
    convert an opencv image to a 3 channel uint8 image
    """
    def __call__(self, image):
        """
        <Args>
        image(numpy array) : Image to be converted to 32-bit floating point

        <Returns>
        image(numy array) : converted image
        """

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)