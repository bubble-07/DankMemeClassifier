"""Lel calm down, it's just a meme classifier, bro"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import listdir

import gc

import argparse
import sys
import tempfile

import random

import scipy as sp
import numpy as np
import tensorflow as tf
import imageio

FLAGS = None

img_width = 128
img_height = 128
img_color_channels = 3
img_vec_size = img_width * img_height * img_color_channels

num_big_memes = 32
num_massive_memes = 1024

micro_meme_size = 16
mixer_meme_size = 32

num_training_iters = 3000

def deepnn(x):
    """deepnn builds the graph for a deep neural net for classifying memes.
        Arguments:
            x: an input tensor with the dimensions (N_examples, img_vec_size)
        Returns:
            A tuple (y, keep_prov). y is a tensor of shape (N_examples, 1)
            with values equal to the probability of classifying
            any given meme as dank
    """
    #First convolutional layer, lel -- map one image to 32 base meme element maps
    #Micro-meme-magic is assumed to occur within a 5x5x3 sliding window
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 3, micro_meme_size])
        b_conv1 = bias_variable([micro_meme_size])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    #Pooling layer - downsamples by 2X
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    #Meme magic mixer -- looking within the next local region (5x5), take all 32
    #feature maps, and recombine into 64 more intelligent meme-element layers
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, micro_meme_size, mixer_meme_size])
        b_conv2 = bias_variable([mixer_meme_size])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        
    #Second pooling layer, same dealio
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    #Memetic mega-macro mixer. This right here is responsible for remembering
    #large-scale structure of memetic objects. Previously, we performed two
    #max pools, so our operable image size is about a fourth of its size in each dim
    #According to some meme theorists (this one), the large-scale structure
    #of a memetic object typically takes up around 1/2 of the image in each
    #dimension, and may appear anywhere within the frame.
    #To determine the output size here, we define a parameter
    #"num_big_memes", which specifies roughly how many large-scale memetic objects
    #need to be remembered to successfully classify memes.
    eigth_width = min(img_width // 8, img_height // 8)
    mega_macro_size = eigth_width
    if (eigth_width % 2 == 0):
        mega_macro_size += 1 #Ensure that this parameter is __odd__, so we get centering
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([mega_macro_size, mega_macro_size, mixer_meme_size, num_big_memes])
        b_conv3 = bias_variable([num_big_memes])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    #Third pooling layer, since we only care about approximate meme positioning
    #If we started with the default width of 128, after this, we should
    #get 16x16 feature maps out of this, which ain't bad.
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    #Great, now using our incredibly high IQ, find the massive memes
    #(image-size meme macros) from all of our big memes. This is a FC layer
    l3InputSize = eigth_width * eigth_width * num_big_memes
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([l3InputSize, num_massive_memes])
        b_fc1 = bias_variable([num_massive_memes])
        h_pool3_flat = tf.reshape(h_pool3, [-1, l3InputSize])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    #Dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #Great, now map all of the massive meme detection maps to a single dank detector
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([num_massive_memes, 2])
        b_fc2 = bias_variable([2])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


meme_base_dir = "B:/memes/resizedmemes/"

dankContent = ['CaseMemesAcademicBeans12317/', 'rBertStrips12317/', 'rBikiniBottomTwitter12317/',
               'rBoneHurtingJuice12317/', 'rDankChristianMemes12317/',
               'rDankMemes12317/', 'rDeepFriedMemes12317/', 'rPrequelMemes12317/',
               'rSurrealMemes12317/', 'rWhoTheFuckUp12317/']
normieContent = ['rAdviceAnimals12317/', 'rComedyCemetery12317/', 'rFellowKids12317/',
                 'rFourPanelCringe12317/', 'rPoliticalHumor12317/', 'rTerribleFacebookMemes12317/']

#Given the directory path, the current vector of attributes
#for every example, the current vector of labels for every example,
#and what to label the stuff loaded from this function,
#load all of the images
def load_images_in_dir(dir_name, x_vec, y_vec, label):
    img_load_count = 0
    for img_name in os.listdir(dir_name):
        img_fullpath = dir_name + img_name
        img_vector = sp.misc.imread(img_fullpath, mode='RGB')
        
        if (img_load_count % 1000 == 0):
            gc.collect()
        img_load_count += 1
        
        x_vec.append(img_vector)
        y_vec.append(label)

def output_memes_to_file(memes, labels, directory):
    for i in range(0, len(memes)):
        meme = memes[i] * 256.0
        label = labels[i]
        true_label = "normie"
        if (label[1] > 0.5):
            true_label = "dank"
        fname = true_label + "_meme_" + str(i) + ".png"
        sp.misc.imsave(directory + fname, meme)

#Loads in all of the meme data, returning two parallel lists, one
#for image data, and the other for labels, and randomizes the lists' ordering
def load_meme_data():
    dank_vector = np.array([0, 1])
    normie_vector = np.array([1, 0])
    x = []
    y = []
    for sub in dankContent:
        meme_subdir = meme_base_dir + sub
        load_images_in_dir(meme_subdir, x, y, dank_vector)
    dank_counter = len(x)
    for sub in normieContent:
        normie_subdir = meme_base_dir + sub
        load_images_in_dir(normie_subdir, x, y, normie_vector)
    normie_counter = len(x) - dank_counter
    print("Dank Memes: ", dank_counter)
    print("Normie Memes: ", normie_counter)
    print("Total Memes: ", len(x))
    #Great, now shuffle both vectors (x and y), maintaining the proper order
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
    return (x, y)

#Given two parallel lists, one for images, and the other for labels,
#returns a list of pairs, where each one has 1/n * origlen
def split_to_folds(x, y, n):
    result = [([], [])] * n
    for i in range(0, len(x)):
        xL, yL = result[i % n]
        xL.append(x[i])
        yL.append(y[i])
    return result

batch_size = 250

#TODO: fix the way that batches are sampled for testing!
def sample_batch(x, y, n):
    x_shape = list(x[0].shape)
    x_shape[:0] = [n]
    y_shape = list(y[0].shape)
    y_shape[:0] = [n]
    x_batch = np.zeros(x_shape)
    y_batch = np.zeros(y_shape)

    #Pick a random __starting position__ so that we get nice caching
    randstart = random.randint(0, len(x) - 1)
    
    for i in range(0, n):
        ind = (i + randstart) % len(x)
        #Doing this to re-normalize the image size
        x_batch[i] = x[ind] / 256.0
        y_batch[i] = y[ind]
    return (x_batch, y_batch)

def flatten(x):
    shape = list(x[0].shape)
    shape[:0] = [len(x)]
    return np.concatenate(x).reshape(shape)

def main(_):
  # Import data
  x_vec, y_vec = load_meme_data()

  print("Memes loaded")

  foldedData = split_to_folds(x_vec, y_vec, 2)

  x_test, y_test = foldedData[0]
  x_train, y_train = foldedData[1]

  print("Memes folded")
  

  # Create the model
  x = tf.placeholder(tf.float32, [None, img_width, img_height, img_color_channels])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  #Is this loss?
  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  gc.collect()

  misclassified_meme_base_dir = "B:/misclassifiedmemes/"


  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_training_iters):
      x_batch, y_batch = sample_batch(x_train, y_train, batch_size)
      if i % 50 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: x_batch, y_: y_batch, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})

    batch_num = int(len(x_test) / batch_size)
    test_accuracy = 0
        
    for i in range(batch_num):
        x_batch, y_batch = sample_batch(x_test, y_test, batch_size)
        test_accuracy += accuracy.eval(feed_dict={x: x_batch,
                                                  y_: y_batch,
                                                  keep_prob: 1.0})

        predictions = correct_prediction.eval(feed_dict={x: x_batch,
                                                  y_: y_batch,
                                                  keep_prob: 1.0})
        misclassedmemes = [i for i in range(0, len(predictions)) if (predictions[i] < 0.5)]
        misclassedimages = x_batch[misclassedmemes]
        misclassedlabels = y_batch[misclassedmemes]
        output_memes_to_file(misclassedimages, misclassedlabels, misclassified_meme_base_dir)


    test_accuracy /= batch_num
    print("test accuracy %g"%test_accuracy)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/B:/memes/resizedmemes',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
        
        
