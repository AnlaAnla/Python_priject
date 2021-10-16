import numpy as np
import matplotlib.pyplot as plt
import mtcnn
import PIL.Image as Image


class getFace:
	def __init__(self):
		self.name = "getFace"
	def load_image(self, filename):
		# load image from file
		image = Image.open(filename)
		# convert to RGB, if needed
		image = image.convert('RGB')
		# convert to array
		pixels = np.array(image)
		return pixels

	# extract the face from a loaded image and resize
	def extract_face(self, model, pixels, required_size=(128, 128)):
		# detect face in the image
		faces = model.detect_faces(pixels)
		# skip cases where we could not detect a face
		if len(faces) == 0:
			return None
		# extract details of the face
		x1, y1, width, height = faces[0]['box']
		# force detected pixel values to be positive (bug fix)
		x1, y1 = abs(x1), abs(y1)
		# convert into coordinates
		x2, y2 = x1 + width, y1 + height
		# retrieve face pixels
		face_pixels = pixels[y1:y2, x1:x2]
		# resize pixels to the model size
		image = Image.fromarray(face_pixels)
		image = image.resize(required_size)
		face_array = np.array(image)
		return face_array

	def load_faces(self, img_path, size):
		model = mtcnn.MTCNN()
		pixels = self.load_image(img_path)
		face = self.extract_face(model, pixels, required_size=size)

		return np.array(face)
