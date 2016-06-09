import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/Users/kevin/Documents/Python/taco/training/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

def initWeight(shape):
    weights = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(weights)

# start with 0.1 so reLu isnt always 0
def initBias(shape):
    bias = tf.constant(0.1,shape=shape)
    return tf.Variable(bias)

# the convolution with padding of 1 on each side, and moves by 1.
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

# max pooling basically shrinks it by 2x, taking the highest value on each feature.
def maxPool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
