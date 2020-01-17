from utils import *
import time
import os
import cv2
import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float

import numpy as np
import tensorflow as tf


# 定义SRCNN类
class SRCNN(object):
	def __init__(self,
				 sess,
				 batch_size=64,
				 checkpoint_dir=None,
				 sample_dir=None):
		self.sess = sess
		self.batch_size = batch_size
		self.checkpoint_dir = checkpoint_dir
		self.sample_dir = sample_dir
		self.build_model()

	# 搭建网络
	def build_model(self):
		self.images = tf.placeholder(tf.float32, [None, 33, 33, 1], name='images')
		self.labels = tf.placeholder(tf.float32, [None, 21, 21, 1], name='labels')
		# 第一层CNN：对输入图片的特征提取。（9 x 9 x 64卷积核）
		# 第二层CNN：对第一层提取的特征的非线性映射（1 x 1 x 32卷积核）
		# 第三层CNN：对映射后的特征进行重建，生成高分辨率图像（5 x 5 x 1卷积核）
		# 权重
		self.weights = {
			# 论文中为提高训练速度的设置 n1=32 n2=16
			'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
			'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
			'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
		}
		# 偏置
		self.biases = {
			'b1': tf.Variable(tf.zeros([64]), name='b1'),
			'b2': tf.Variable(tf.zeros([32]), name='b2'),
			'b3': tf.Variable(tf.zeros([1]), name='b3')
		}
		self.pred = self.model()
		# 以MSE作为损耗函数
		self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
		self.saver = tf.train.Saver()

	def train(self, config):

		prepare_for_train(self.sess)
		data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
		train_data, train_label = read_data(data_dir)
		global_step = tf.Variable(0)

		learning_rate_exp = tf.train.exponential_decay(config.learning_rate, global_step, 1480, 0.98,
													   staircase=True)  # 每1个Epoch 学习率*0.98
		self.train_op = tf.train.GradientDescentOptimizer(learning_rate_exp).minimize(self.loss,
																					  global_step=global_step)
		tf.global_variables_initializer().run()
		start_time = time.time()
		res = self.load(self.checkpoint_dir)
		counter = 0
		if res:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
		print("Training...")
		for ep in range(config.epoch):
			# 以batch为单元
			batch_idxs = len(train_data) // config.batch_size
			for idx in range(0, batch_idxs):
				batch_images = train_data[idx * config.batch_size: (idx + 1) * config.batch_size]
				batch_labels = train_label[idx * config.batch_size: (idx + 1) * config.batch_size]
				counter += 1
				_, err = self.sess.run([self.train_op, self.loss],
									   feed_dict={self.images: batch_images, self.labels: batch_labels})
				if counter % 10 == 0:  # 10的倍数step显示
					print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
						  % ((ep + 1), counter, time.time() - start_time, err))
				if counter % 500 == 0:  # 500的倍数step存储
					self.save(config.checkpoint_dir, counter)

	def test(self, path, config):
		tf.global_variables_initializer().run()
		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
		data, label, color, nx, ny = parpare_for_test(self.sess, path)
		conv_out = self.pred.eval({self.images: data})
		conv_out = merge(conv_out, [nx, ny])
		conv_out = conv_out.squeeze()
		result_bw = revert(conv_out)
		image_path = os.path.join(os.getcwd(), config.sample_dir)
		image_path = os.path.join(image_path, "MySRCNN_simple.bmp")
		imsave(result_bw, image_path)
		result = np.zeros([result_bw.shape[0], result_bw.shape[1], 3], dtype=np.uint8)
		result[:, :, 0] = result_bw
		result[:, :, 1:3] = color
		result = cv.cvtColor(result, cv.COLOR_YCrCb2RGB)
		image_dir = os.path.join(os.getcwd(), config.sample_dir)
		image_path = os.path.join(image_dir, "MySRCNN.bmp")
		imsave(result, image_path)

		conv_out = merge(label, [nx, ny])
		conv_out = conv_out.squeeze()
		result_bw = revert(conv_out)
		result = np.zeros([result_bw.shape[0], result_bw.shape[1], 3], dtype=np.uint8)
		result[:, :, 0] = result_bw
		result[:, :, 1:3] = color
		bicubic = cv.cvtColor(result, cv.COLOR_YCrCb2RGB)
		bicubic_path = os.path.join(image_dir, 'Orig_MySRCNN.bmp')
		imsave(bicubic, bicubic_path)

	def upscaling(self, path, config, scale):
		tf.global_variables_initializer().run()
		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
		data, label, color, nx, ny = parpare_for_upscaling(self.sess, path, scale)
		conv_out = self.pred.eval({self.images: data})
		conv_out = merge(conv_out, [nx, ny])
		result_bw = revert(conv_out)
		result_bw = result_bw.squeeze()
		result = np.zeros([result_bw.shape[0], result_bw.shape[1], 3], dtype=np.uint8)
		result[:, :, 0] = result_bw
		result[:, :, 1:3] = color
		result = cv.cvtColor(result, cv.COLOR_YCrCb2RGB)
		image_dir = os.path.join(os.getcwd(), config.sample_dir)
		image_path = os.path.join(image_dir, "MySRCNN.bmp")
		imsave(result, image_path)

		conv_out = merge(label, [nx, ny])
		conv_out = conv_out.squeeze()
		result_bw = revert(conv_out)
		result = np.zeros([result_bw.shape[0], result_bw.shape[1], 3], dtype=np.uint8)
		result[:, :, 0] = result_bw
		result[:, :, 1:3] = color
		orig = cv.cvtColor(result, cv.COLOR_YCrCb2RGB)
		bicubic_path = os.path.join(image_dir, 'Orig_MySRCNN.bmp')
		imsave(orig, bicubic_path)

	def model(self):
		conv1 = tf.nn.relu(
			tf.nn.conv2d(self.images, self.weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b1'])
		conv2 = tf.nn.relu(
			tf.nn.conv2d(conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b2'])
		conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b3']
		return conv3

	def save(self, checkpoint_dir, step):
		model_name = "SRCNN.model"
		model_dir = "%s_%s" % ("srcnn", 21)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)  # 再一次确定路径为 checkpoint->srcnn_21下
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),  # 文件名为SRCNN.model-迭代次数
						global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoints...")
		model_dir = "%s_%s" % ("srcnn", 21)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)  # 路径为checkpoint->srcnn_labelsize(21)
		# 加载路径下的模型(.meta文件保存当前图的结构; .index文件保存当前参数名; .data文件保存当前参数值)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir,
													   ckpt_name))  # saver.restore()函数给出model.-n路径后会自动寻找参数名-值文件进行加载

			return True, ckpt_name[12:]
		else:
			return False

	def action(self, config):
		if config.action == 1:
			self.train(config)
		elif config.action == 2:
			self.test('/home/chengcongyue/PycharmProjects/SRCNN12/Test/Set5/butterfly_GT.bmp', config)
		elif config.action == 3:
			self.upscaling('/home/chengcongyue/PycharmProjects/SRCNN12/Test/Set5/butterfly_GT.bmp', config, 3)
