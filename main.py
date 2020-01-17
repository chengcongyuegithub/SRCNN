from model import SRCNN
import tensorflow as tf
import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 2000, "训练多少波")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_float("learning_rate", 1e-2, "学习率")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "checkpoint directory名字")
flags.DEFINE_string("sample_dir", "sample", "sample directory名字")
flags.DEFINE_integer("action", 1, "1:训练,2:测试,3:放大")
FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()


def main(_):
	pp.pprint(flags.FLAGS.__flags)
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)
	with tf.Session() as sess:
		srcnn = SRCNN(sess,
					  checkpoint_dir=FLAGS.checkpoint_dir,
					  sample_dir=FLAGS.sample_dir)
		srcnn.action(FLAGS)
if __name__ == '__main__':
	tf.app.run()