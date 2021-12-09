import face_recognition
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

import util
from model import Model


def blur(img, x, y, w, h):
    updated = img.copy()
    blurred = cv.GaussianBlur(updated, (75, 75), 0)
    # Swap x y for image indexing
    updated[y:y+h, x:x+w] = blurred[y:y+h, x:x+w]
    return updated


class HOG_SVM(Model):

    def __init__(self):
        super().__init__()
        self.model = None
        self.images = None

    def predict(self):
        i = 0
        while i < len(self.images):
            # Detect boxes and save to Image
            boxes = self.model.face_locations(self.images[i].original)
            self.images[i].bounding_boxes = boxes

            if self.images[i].bounding_boxes != ():
                print(i, ": Predicted", self.images[i].bounding_boxes)
                print(i, ": True Box ", self.images[i].true_boxes)

                # Add blur to image for each detected box
                blur_img = self.images[i].original.copy()
                for detected_box in self.images[i].bounding_boxes:
                    a1, a2, a3, a4 = detected_box
                    blur_img = blur(blur_img, a1, a2, a3, a4)

                # Save blurred image on Image
                self.images[i].blurred = blur_img

                # Display image
                cv.imshow("Blurred image", self.images[i].blurred)
                cv.waitKey(0)

                # Break after 10 images for testing
                if i > 10:
                    break
            i += 1
