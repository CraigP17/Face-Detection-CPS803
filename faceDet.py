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
		self.model = None
		self.images = None

	def train(self):
		self.model = cv.dnn.readNetFromCaffe(configFile, modelFile)

	def predict(self):
		sample_result = True    # Show the model result for 2 images
		if sample_result:
			self.images = self.read_sample_images()

		for img in self.images:
			im_blob = cv.dnn.blobFromImage(cv.resize(img.original, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
			self.model.setInput(im_blob)
			img.boxes = self.model.forward()

			if sample_result:
				# Show original image with true box
				ox = img.true_boxes[0][0]
				oy = img.true_boxes[0][1]
				ox1 = img.true_boxes[0][2]
				oy1 = img.true_boxes[0][3]
				cv.imshow('Original', img.original)
				cv.waitKey(0)
				og = cv.rectangle(img.original, (ox,oy),(ox+ox1,oy+oy1),(0,0,255),3)
				# cv.imshow('Original', og)
				# cv.waitKey(0)
				pr = img.original
				for detected_box in img.boxes:
					# Show image with predicted bounding box
					(x, y, x1, y1) = detected_box.astype("int")
					pr = cv.rectangle(img.original, (x,y),(x+x1,y+y1),(0,255,0),3)
				cv.imshow('Predicted', pr)
				cv.waitKey(0)
				# Show image after blurring
				cv.imshow('Blurred', img.blurred)
				cv.waitKey(0)

	def draw_faces(self):
		for im in self.images:
			h, w = im.original.shape[:2]
			confidence = im.boxes[0, 0, 0, 2]
			if confidence > 0.95:
				box = im.boxes[0, 0, 0, 3:7] * np.array([w, h, w, h])
				(x, y, x1, y1) = box.astype("int")
				im.blurred = blur(im.original, x, y, x1, y1)


