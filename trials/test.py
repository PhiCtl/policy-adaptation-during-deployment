import cv2
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import itertools
from ast import literal_eval
from src.env.dm_control.dm_control import mujoco

def color_jitter(x, params) :

	assert isinstance(x, np.ndarray), 'inputs must be numpy arrays'
	assert x.dtype == np.uint8, 'inputs must be uint8 arrays'
	x = np.moveaxis(np.array(x), -1, 0)[:3]
	im = TF.to_pil_image(torch.ByteTensor(x))
	# jitter
	img = TF.adjust_brightness(im, params["b"])
	img = TF.adjust_contrast(img,1.5)
	img = TF.adjust_hue(img, params["h"])
	#out = np.moveaxis(np.array(img), -1, 0)[:3]

	return np.array(img)


def compute_distance(img1, img2):
	print(img1.shape)
	img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
	img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

	x1 = np.cos(img1_hsv[:, :, 0] * np.pi / 180) * img1_hsv[:, :, 1]
	y1 = np.sin(img1_hsv[:, :, 0] * np.pi / 180) * img1_hsv[:, :, 1]
	z1 = img1_hsv[:, :, 2]

	x2 = np.cos(img2_hsv[:, :, 0] * np.pi / 180) * img2_hsv[:, :, 1]
	y2 = np.sin(img2_hsv[:, :, 0] * np.pi / 180) * img2_hsv[:, :, 1]
	z2 = img2_hsv[:, :, 2]

	return np.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2).sum())

def test_mujoco():


def main() :

	hue = [0.1, 0.2, 0.3, 0.4, 0.5]
	contrast = [0.5, 1.5]
	brightness = [0.5, 1.5]
	combinations = list(itertools.product(hue, contrast, brightness))


	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	changes = []

	for i in range(5) :
		path_img = "src/env/data/video" + str(i) + "_frame.jpeg"
		img = cv2.imread(path_img)
		for h, c, b in combinations:
			img2 = color_jitter(img, {"h" : h, "c" : c, "b" : b})
			changes.append({"background" : path_img[-17:-5] , \
							"params" : {"h" : h, "c" : c, "b" : b} , \
							"distance" : compute_distance(img, img2)})

	df = pd.DataFrame(changes)
	for name, grp in df.sort_values("distance", ascending = True).groupby("background") :
		path_data = "../Results/" + name + ".csv"
		grp.to_csv(path_data)

def test() :
	path_img = "src/env/data/video0_frame.jpeg"
	path_data = "../Results/video0_frame.csv"
	img = cv2.imread(path_img)

	data = pd.read_csv(path_data, index_col = 0, converters = {"params" : literal_eval})#.sort_values("mean", ascending = False)
	changes = data["params"].values
	print(data["distance"])

	# for i, change in enumerate(changes) :
	# 	img2 = color_jitter(img, change)
	# 	cv2.imshow("fig", img2)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()
	# 	print("{}, {}".format(i, change))

def introduce_object():

	path_img = "../Results/bkg/test.jpeg"
	path_tr = path_img[:-5] + "_object.jpeg"
	img = cv2.imread(path_img)
	cv2.circle(img, (350, 350), 20, (0,0,255, -1))
	cv2.imshow("fig", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	#cv2.imwrite(path_tr, out)

if __name__ == "__main__":
	test_mujoco()


