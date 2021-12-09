import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

import util
from model import Model
from image import Image

valid_path_pos = 'validation_set_positives'
valid_path_neg = 'validation_set_negatives'
modelFile = "models/opencv_face_detector.caffemodel"
configFile = "models/deploy.prototxt"


def blur(img, x, y, w, h):
	updated = img.copy()
	blurred = cv.GaussianBlur(updated, (75, 75), 0)
	updated[y:h, x:w] = blurred[y:h, x:w]
	return updated


class FaceDetDNN(Model):
	def __init__(self):
		super().__init__()
		self.model = cv.dnn.readNetFromCaffe(configFile, modelFile)
		self.images = None

	def predict(self):
		for img in self.images:
			im_blob = cv.dnn.blobFromImage(cv.resize(img.original, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
			self.model.setInput(im_blob)
			img.boxes = self.model.forward()

	def draw_faces(self):
		for im in self.images:
			h, w = im.original.shape[:2]
			confidence = im.boxes[0, 0, 0, 2]
			if confidence > 0.95:
				box = im.boxes[0, 0, 0, 3:7] * np.array([w, h, w, h])
				(x, y, x1, y1) = box.astype("int")
				im.blurred = blur(im.original, x, y, x1, y1)
				cv.imshow("face", im.blurred)
				cv.waitKey(0)


paths = [valid_path_pos, valid_path_neg]
# paths = [valid_path_neg]
# paths = [valid_path_pos]
myfaceDet = FaceDetDNN()
myfaceDet.read_preprocess()
myfaceDet.predict()
myfaceDet.draw_faces()
