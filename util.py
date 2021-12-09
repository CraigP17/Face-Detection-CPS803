import numpy as np
from shapely.geometry import Polygon

def calcError(images):
	for im in images:
		print(im)
		print(im.boxes)
		print(im.trueBox)
		box = calculate_box(im.boxes)
		trueBox = calculate_box(im.trueBox)
		im.error = iou(box,trueBox)
		print(im.error)

def calculate_box(arr):
	x,y,w,h = arr
	bottom_left = [x,y]
	top_left = [x,y+h]
	top_right = [x+w,y+h]
	bottom_right = [x+w,y]
	return [top_left, top_right, bottom_right, bottom_left]

def iou(box1, box2):
	p1 = Polygon(box1)
	p2 = Polygon(box2)
	ret = p1.intersection(p2).area / p1.union(p2).area
	return ret 
	

