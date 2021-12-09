import cv2 as cv
import matplotlib.pyplot as plt
from time import sleep

import util
from model import Model


def blur(img, x, y, w, h):
    updated = img.copy()
    blurred = cv.GaussianBlur(updated, (75, 75), 0)
    # Swap x y for image indexing
    updated[y:y+h, x:x+w] = blurred[y:y+h, x:x+w]
    return updated


class HaarCascades(Model):

    def __init__(self):
        super().__init__()
        self.model = None
        self.images = None

    def train(self):
        harr_model_path = 'models/haarcascade_frontalface_default.xml'
        face_cascade = cv.CascadeClassifier(harr_model_path)
        self.model = face_cascade

    def predict(self):
        i = 0
        while i < len(self.images):
            # Detect boxes and save to Image
            gray_img = cv.cvtColor(self.images[i].original, cv.COLOR_BGR2GRAY)
            boxes = self.model.detectMultiScale(gray_img,
                                                scaleFactor=1.3,
                                                minNeighbors=5)
            self.images[i].bounding_boxes = boxes

            if self.images[i].bounding_boxes != ():
                print(i, ": Predicted", self.images[i].bounding_boxes)
                print(i, ": True Box ", self.images[i].true_boxes)

                # Add blur to image for each detected box
                blur_img = self.images[i].original.copy()
                for detected_box in self.images[i].bounding_boxes:
                    x, y, w, h = detected_box
                    blur_img = blur(blur_img, x, y, w, h)

                # Save blurred image on Image
                self.images[i].blurred = blur_img

                # Show detected box on image
                for box in self.images[i].bounding_boxes:
                    x, y, w, h = box
                    cv.rectangle(self.images[i].original, (x, y),
                                 (x + w, y + h), (0, 255, 0), 3)
                plt.imshow(cv.cvtColor(self.images[i].original, cv.COLOR_BGR2RGB))

                # Display image
                cv.imshow('blurred', self.images[i].blurred)
                cv.waitKey(0)

                # Break after 10 images for testing
                if i > 10:
                    break
            i += 1
