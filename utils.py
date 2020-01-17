import numpy as np
import os
import cv2 as cv
import h5py
import scipy.misc
import scipy.ndimage
import glob
import tensorflow as tf


# 保存图片
def imsave(image, path):
	return scipy.misc.imsave(path, image)


# 读取图片
def imread(path, is_grayscale=True):
	# 读指定路径的图像
	if is_grayscale:
		return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
	else:
		return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


# 读取h5
def read_data(path):
	with h5py.File(path, 'r') as hf:  # 读取h5格式数据文件(用于训练或测试)
		data = np.array(hf.get('data'))
		label = np.array(hf.get('label'))
		return data, label


# 裁剪图片1
def modcrop_small(image):
	padding2 = 6
	if len(image.shape) == 3:
		h, w, _ = image.shape
		h = (h - 33 + 1) // 21 * 21 + 21 + padding2
		w = (w - 33 + 1) // 21 * 21 + 21 + padding2
		image1 = image[padding2:h, padding2:w, :]  # 6
	else:
		h, w = image.shape
		h = (h - 33 + 1) // 21 * 21 + 21 + padding2
		w = (w - 33 + 1) // 21 * 21 + 21 + padding2
		image1 = image[padding2:h, padding2:w]
	return image1


# 裁剪图片2
def modcrop(image, scale=3):
	# 把图像的长和宽都变成scale的倍数
	if len(image.shape) == 3:
		h, w, _ = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w, :]
	else:
		h, w = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w]
	return image


# 子图合成原图
def merge(images, size):
	h, w = images.shape[1], images.shape[2]  # 觉得下标应该是0,1
	# h, w = images.shape[0], images.shape[1]
	img = np.zeros((h * size[0], w * size[1], 1))
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[j * h:j * h + h, i * w:i * w + w, :] = image
	return img


# 归一化和还原
def im2double(im):
	info = np.iinfo(im.dtype)
	return im.astype(np.float) / info.max


def revert(im):
	im = im * 255
	im[im > 255] = 255
	im[im < 0] = 0
	return im.astype(np.uint8)


def preprocess_for_train(path, scale=3):
	scale -= 1
	image = imread(path, is_grayscale=True)
	label_ = modcrop(image, scale)

	# Must be normalized
	image = image / 255.
	label_ = label_ / 255.

	input_ = scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
	input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)

	return input_, label_


def prepare_for_train(sess):
	data_dir = os.path.join(os.getcwd(), "Train")
	data = glob.glob(os.path.join(data_dir, "*.bmp"))
	print(len(data))  #
	sub_input_sequence = []
	sub_label_sequence = []
	padding = 6
	for i in range(len(data)):
		input_, label_ = preprocess_for_train(data[i], 3)
		if len(input_.shape) == 3:
			h, w, _ = input_.shape
		else:
			h, w = input_.shape
		for x in range(0, h - 33 + 1, 14):
			for y in range(0, w - 33 + 1, 14):
				sub_input = input_[x:x + 33, y:y + 33]
				sub_label = label_[x + 6:x + 6 + 21,
							y + 6:y + 6 + 21]
				sub_input = sub_input.reshape([33, 33, 1])
				sub_label = sub_label.reshape([21, 21, 1])
				sub_input_sequence.append(sub_input)
				sub_label_sequence.append(sub_label)
	# 上面的部分和训练是一样的
	arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
	arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]
	savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
	with h5py.File(savepath, 'w') as hf:
		hf.create_dataset('data', data=arrdata)
		hf.create_dataset('label', data=arrlabel)


def preprocess_for_test(path, scale=3):
	scale -= 1
	im = cv.imread(path)
	im = cv.cvtColor(im, cv.COLOR_BGR2YCR_CB)
	img = im2double(im)

	label_ = modcrop(img, scale=scale)
	color_base = modcrop(im, scale=scale)
	label_ = label_[:, :, 0]
	input_ = scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
	input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)

	label_small = modcrop_small(label_)  # 把原图裁剪成和输出一样的大小
	input_small = modcrop_small(input_)  # 把原图裁剪成和输出一样的大小
	color_small = modcrop_small(color_base[:, :, 1:3])
	imsave(input_small, '/home/chengcongyue/PycharmProjects/SRCNN12/sample/input_.bmp')
	imsave(label_small, '/home/chengcongyue/PycharmProjects/SRCNN12/sample/label_.bmp')

	return input_, label_, color_small


def parpare_for_test(sess, path):
	sub_input_sequence = []
	sub_label_sequence = []
	padding = 6
	# 测试
	input_, label_, color = preprocess_for_test(path, 3)  # 测试图片
	if len(input_.shape) == 3:
		h, w, _ = input_.shape
	else:
		h, w = input_.shape
	nx = 0
	ny = 0
	for x in range(0, h - 33 + 1, 21):
		nx += 1
		ny = 0
		for y in range(0, w - 33 + 1, 21):
			ny += 1
			sub_input = input_[x:x + 33, y:y + 33]
			sub_label = label_[x + 6:x + 6 + 21,
						y + 6:y + 6 + 21]
			sub_input = sub_input.reshape([33, 33, 1])
			sub_label = sub_label.reshape([21, 21, 1])
			sub_input_sequence.append(sub_input)
			sub_label_sequence.append(sub_label)
	color = np.array(color)
	data = np.asarray(sub_input_sequence)
	label = np.asarray(sub_label_sequence)
	return data, label, color, nx, ny


def preprocess_for__upscaling(path, scale=3):
	im = cv.imread(path)
	size = im.shape
	im = scipy.misc.imresize(im, [size[0] * scale, size[1] * scale], interp='bicubic')
	im = cv.cvtColor(im, cv.COLOR_BGR2YCR_CB)
	img = im2double(im)
	label_ = modcrop(img, scale=scale)
	color_base = modcrop(im, scale=scale)
	label_ = label_[:, :, 0]

	label_small = modcrop_small(label_)  # 把原图裁剪成和输出一样的大小
	color_small = modcrop_small(color_base[:, :, 1:3])
	imsave(label_small, '/home/chengcongyue/PycharmProjects/SRCNN5/sample/label_.bmp')

	return label_, label_, color_small


def parpare_for_upscaling(sess, path, scale):
	sub_input_sequence = []
	sub_label_sequence = []
	padding = 6
	# 测试
	input_, label_, color = preprocess_for__upscaling(path, scale)  # 测试图片
	if len(input_.shape) == 3:
		h, w, _ = input_.shape
	else:
		h, w = input_.shape
	nx = 0  # 后注释
	ny = 0  # 后注释
	# 自图需要进行合并操作
	for x in range(0, h - 33 + 1, 21):  # x从0到h-33+1 步长stride(21)
		nx += 1
		ny = 0
		for y in range(0, w - 33 + 1, 21):  # y从0到w-33+1 步长stride(21)
			ny += 1
			sub_input = input_[x:x + 33, y:y + 33]  # [33 x 33]
			sub_label = label_[x + 6:x + 6 + 21,
						y + 6:y + 6 + 21]  # [21 x 21]
			sub_input = sub_input.reshape([33, 33, 1])
			sub_label = sub_label.reshape([21, 21, 1])
			sub_input_sequence.append(sub_input)
			sub_label_sequence.append(sub_label)
	color = np.array(color)
	data = np.asarray(sub_input_sequence)
	label = np.asarray(sub_label_sequence)
	return data, label, color, nx, ny
