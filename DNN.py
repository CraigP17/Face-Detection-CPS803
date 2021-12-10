import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

import util
from model import Model
from image import Image

modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "models/deploy.prototxt"


def blur(img, x, y, w, h):
	updated = img.copy()
	blurred = cv.GaussianBlur(updated, (75, 75), 0)
	updated[y:h, x:w] = blurred[y:h, x:w]
	return updated


class DNN(Model):
	def __init__(self):
		super().__init__()
		self.model = None
		self.images = None

	def train(self):
		self.model = cv.dnn.readNetFromCaffe(configFile, modelFile)

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
				# show = cv.rectangle(im.blurred, (x, y),(x1, y1), (0, 255, 0), 3)
				# a,b,c,d = im.true_boxes[0]
				# show = cv.rectangle(im.blurred, (a,b),(a+c,b+d),(255,0,0),3)
				# print(box)
				# print(im.true_boxes)
				# cv.imshow('lines',show)
				# cv.waitKey(0)
		
