import pandas as pd
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

valid_path_pos = 'validation_set_positives'
valid_path_neg = 'validation_set_negatives'
modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "models/deploy.prototxt"
class Image:
	def __init__(self, img):
		self.original = cv.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
		self.blurred = []
		self.boxes = []

def blur(img, x, y, w, h):
	updated = img.copy()
	blurred = cv.GaussianBlur(updated,(75,75),0)
	updated[y:h,x:w] = blurred[y:h,x:w]
	return updated

class DNN:
	def __init__(self):
		self.model = cv.dnn.readNetFromCaffe(configFile, modelFile)
		self.images = None

	def read_preprocess(self, list_paths):
		images = []
		for path in list_paths:
			files = os.listdir(path)
			for file_name in files:
				image = cv.imread(f"{path}/{file_name}")
				img_obj = Image(image)
				images.append(img_obj)
		self.images = images

	def predict2(self):
		print("123")

	def predict(self):
		for img in self.images:
			im_blob = cv.dnn.blobFromImage(cv.resize(img.original,(300,300)), 1.0, (300,300), (104.0, 117.0, 123.0))
			self.model.setInput(im_blob)
			img.boxes = self.model.forward()


	def drawFaces(self):
		for im in self.images:
			h,w = im.original.shape[:2]
			confidence = im.boxes[0, 0, 0, 2]
			if confidence > 0.95:
				box = im.boxes[0, 0, 0, 3:7] * np.array([w,h, w,h])
				(x, y, x1, y1) = box.astype("int")
				im.blurred = blur(im.original, x,y,x1,y1)
				cv.imshow("face", im.blurred)
				cv.waitKey(0)



paths = [valid_path_pos,valid_path_neg]
#paths = [valid_path_neg]
#paths = [valid_path_pos]
myDNN = DNN()
myDNN.read_preprocess(paths)
myDNN.predict()
myDNN.drawFaces()