import os
import cv2 as cv

import preprocess
import util
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
                image = cv.imread(filepath)
                img_obj = Image(image, true_boxes=images_dict[filepath])
                images.append(img_obj)
        self.images = images

    def evaluate(self):
        util.calcError(self.images)
        print("Calculating mean average precision")
        print("map: ", util.mAP(self.images))
        print("Calculating average coverage")
        print("avg: ", util.coverageAccuracy(self.images))
        print("Calculating false positive rate")
        print("fp: ", util.fpr(self.images))
        print()
        print()
        return util.mAP(self.images), util.coverageAccuracy(self.images), util.fpr(self.images)
