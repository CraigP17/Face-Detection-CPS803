import os
import cv2 as cv

import preprocess
from image import Image

class Model:
    def __init__(self):
        self.model = None
        self.images = None

    def read_preprocess(self):
        images_dict = preprocess.read_image_data()
        images = []
        for filepath in images_dict:
            if os.path.isfile(filepath):
                # Ensure image file exists
                image = cv.read(filepath)
                img_obj = Image(image, trueBoxes=images_dict[filepath])
                img_obj.trueBoxes
                images.append(img_obj)
        self.images = images
