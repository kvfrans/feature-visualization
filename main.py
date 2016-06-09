import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/Users/kevin/Documents/Python/feature-visualization/training/train',
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


batchsize = 50;
imagesize = 32;
colors = 3;

def train():
    sess = tf.InteractiveSession()

    img = tf.placeholder("float",shape=[None,imagesize*imagesize*colors])
    lbl = tf.placeholder("float",shape=[None,10])
    xImage = tf.reshape(img,[-1,imagesize,imagesize,colors])
    # for each 5x5 area, check for 32 features over 3 color channels
    wConv1 = initWeight([5,5,colors,32])
    bConv1 = initBias([32])
    # move the conv filter over the picture
    conv1 = conv2d(xImage,wConv1)
    # adds bias
    bias1 = conv1 + bConv1
    # relu = max(0,x), adds nonlinearality
    relu1 = tf.nn.relu(bias1)
    # maxpool to 16x16
    pool1 = maxPool2d(relu1)
    # second conv layer, takes a 16x16 with 32 layers, turns to 8x8 with 64 layers
    wConv2 = initWeight([5,5,32,64])
    bConv2 = initBias([64])
    conv2 = conv2d(pool1,wConv2)
    bias2 = conv2 + bConv2
    relu2 = tf.nn.relu(bias2)
    pool2 = maxPool2d(relu2)
    # fully-connected is just a regular neural net: 8*8*64 for each training data
    wFc1 = initWeight([(imagesize/4) * (imagesize/4) * 64, 1024])
    bFc1 = initBias([1024])
    # reduce dimensions to flatten
    pool2flat = tf.reshape(pool2, [-1, (imagesize/4) * (imagesize/4) *64])
    # 128 training set by 2304 data points
    fc1 = tf.matmul(pool2flat,wFc1) + bFc1;
    relu3 = tf.nn.relu(fc1);
    # dropout removes duplicate weights
    keepProb = tf.placeholder("float");
    drop = tf.nn.dropout(relu3,keepProb);
    wFc2 = initWeight([1024,10]);
    bFc2 = initWeight([10]);
    # softmax converts individual probabilities to percentages
    guesses = tf.nn.softmax(tf.matmul(drop, wFc2) + bFc2);
    # how wrong it is
    cross_entropy = -tf.reduce_sum(lbl*tf.log(guesses + 1e-9));
    # theres a lot of tensorflow optimizers such as gradient descent
    # adam is one of them
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy);
    # array of bools, checking if each guess was correct
    correct_prediction = tf.equal(tf.argmax(guesses,1), tf.argmax(lbl,1));
    # represent the correctness as a float [1,1,0,1] -> 0.75
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"));


    sess.run(tf.initialize_all_variables());

    batch = unpickle("cifar-10-batches-py/data_batch_1")

    validationData = batch["data"][555:batchsize+555]
    validationRawLabel = batch["labels"][555:batchsize+555]
    validationLabel = np.zeros((batchsize,10))
    validationLabel[np.arange(batchsize),validationRawLabel] = 1
    validationData = tnpimg/255.0

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint("/Users/kevin/Documents/Python/taco/training/"))

    # train for 20000
    mnistbatch = mnist.train.next_batch(batchsize)
    # print mnistbatch[0].shape
    for i in range(20000):
        randomint = randint(0,10000 - batchsize - 1)
        trainingData = batch["data"][randomint:batchsize+randomint]
        rawlabel = batch["labels"][randomint:batchsize+randomint]
        trainingLabel = np.zeros((batchsize,10))
        trainingLabel[np.arange(batchsize),rawlabel] = 1
        trainingData = trainingData/255.0

        if i%10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
            img: validationData, lbl: validationLabel, keepProb: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))

            if i%50 == 0:
                saver.save(sess, FLAGS.train_dir, global_step=i)

        optimizer.run(feed_dict={img: trainingData, lbl: trainingLabel, keepProb: 0.5})
        print i


def main(argv=None):
    train()
