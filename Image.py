class Image:
	def __init__(self, img):
		self.original = cv.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
		self.blurred = []
		self.boxes = []
		self.trueBox = []
		self.error = 0
