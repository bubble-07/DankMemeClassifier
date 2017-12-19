"""Lel calm down, it's just a meme GAN, bro"""

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

#Parameters related to image size/number of color channels
img_width = 128
img_height = 128
img_color_channels = 3
img_vec_size = img_width * img_height * img_color_channels

#Parameters related to the discriminator
num_big_memes = 32
num_massive_memes = 1024

micro_meme_size = 16
mixer_meme_size = 32

#Training iteration cap before termination
num_training_iters = 10000

#Parameters related to the generator
gen_seed_size = 30
num_highlevel_concepts = 1024
galactic_concept_width = 16
galactic_concept_depth = 128
starsystem_concept_width = 32
starsystem_concept_depth = 128
planetary_concept_width = 64
planetary_concept_depth = 128
continental_concept_depth = 32

gen_l1inputsize = gen_seed_size + 2
gen_l2outputsize = (galactic_concept_width ** 2) * galactic_concept_depth



#Returns the list of tunable parameters in the generative model
#in the order [weights_l1, biases_l1, weights_l2, ...]
def gennn_params():
    #W_fc1 = weight_variable([gen_l1inputsize, gen_l2outputsize])
    #b_fc1 = bias_variable([gen_l2outputsize])
    
    W_fc1 = weight_variable([gen_l1inputsize, img_vec_size])
    b_fc1 = bias_variable([img_vec_size])

    W_deconv1 = weight_variable([9, 9, starsystem_concept_depth, galactic_concept_depth])
    b_deconv1 = bias_variable([starsystem_concept_depth])

    W_deconv2 = weight_variable([7, 7, planetary_concept_depth, starsystem_concept_depth])
    b_deconv2 = bias_variable([planetary_concept_depth])
    
    W_deconv3 = weight_variable([5, 5, continental_concept_depth, planetary_concept_depth])
    b_deconv3 = bias_variable([continental_concept_depth])

    W_conv1 = weight_variable([5, 5, continental_concept_depth, img_color_channels])
    b_conv1 = bias_variable([img_color_channels])

    return [W_fc1, b_fc1, W_deconv1, b_deconv1, W_deconv2, b_deconv2, W_deconv3, b_deconv3, W_conv1, b_conv1]

#Definition of the __generator__ neural network,
#which here is responsible for generating counterfeit (meme, label)
#pairs, given some kind of random "seed" vector
#and a hard-coded "meme class" vector y
def gennn(seed, y, param_vec):
    #First, load in all of the tunable parameters of the generator
    [W_fc1, b_fc1, W_deconv1, b_deconv1, W_deconv2, b_deconv2, W_deconv3, b_deconv3, W_conv1, b_conv1] = param_vec
    
    #First, append the meme class to the initial generator vector
    with tf.name_scope('concat1'):
        concat_seed = tf.concat([seed, y], 1)
    #First layer here could be responsible for translating the high-level
    #space-independent random seeed into spatially-aware concepts
    #with tf.name_scope('fc1'):
    #    h_fc1 = tf.reshape(tf.nn.leaky_relu(tf.matmul(concat_seed, W_fc1) + b_fc1),
     #                      [batch_size, galactic_concept_width, galactic_concept_width, galactic_concept_depth])
     
    with tf.name_scope('fc1'):
        h_fc1 = tf.reshape(tf.nn.leaky_relu(tf.matmul(concat_seed, W_fc1) + b_fc1),
                           [batch_size, img_width, img_height, img_color_channels])
     
    #The rest of the layers here are responsible for "filling in" the details
    #with tf.name_scope('deconv1'):
    #    out_dims_deconv1 = [batch_size, starsystem_concept_width, starsystem_concept_width, starsystem_concept_depth]
    #    h_deconv1 = tf.nn.leaky_relu(deconv2d(h_fc1, W_deconv1, out_dims_deconv1) + b_deconv1)

    #with tf.name_scope('deconv2'):
    #    out_dims_deconv2 = [batch_size, planetary_concept_width, planetary_concept_width, planetary_concept_depth]
    #    h_deconv2 = tf.nn.leaky_relu(deconv2d(h_deconv1, W_deconv2, out_dims_deconv2) + b_deconv2)

    #Final deconvolutional layer
    #with tf.name_scope('deconv3'):
    #    out_dims_deconv3 = [batch_size, img_width, img_height, continental_concept_depth]
    #    h_deconv3 = tf.nn.leaky_relu(deconv2d(h_deconv2, W_deconv3, out_dims_deconv3) + b_deconv3)

    #Convolutional layer to smooth things over
    #with tf.name_scope('conv1'):
    #    h_conv1 = tf.nn.leaky_relu(conv2d(h_deconv3, W_conv1) + b_conv1)

    
    #Now, our output is the pairing of this big new image together with the meme class label
    return (h_fc1, y)

eigth_width = min(img_width // 8, img_height // 8)
mega_macro_size = eigth_width
if (eigth_width % 2 == 0):
    mega_macro_size += 1 #Ensure that this parameter is __odd__, so we get centering
l3InputSize = eigth_width * eigth_width * num_big_memes


def discrimnn_params():
    W_conv1 = weight_variable([5, 5, img_color_channels, micro_meme_size])
    b_conv1 = bias_variable([micro_meme_size])

    W_conv2 = weight_variable([5, 5, micro_meme_size, mixer_meme_size])
    b_conv2 = bias_variable([mixer_meme_size])

    W_conv3 = weight_variable([mega_macro_size, mega_macro_size, mixer_meme_size, num_big_memes])
    b_conv3 = bias_variable([num_big_memes])

    W_fc1 = weight_variable([l3InputSize, num_massive_memes])
    b_fc1 = bias_variable([num_massive_memes])

    W_fc2 = weight_variable([num_massive_memes, 8])
    b_fc2 = bias_variable([8])
    
    W_fc3 = weight_variable([10, 5])
    b_fc3 = bias_variable([5])

    W_fc41 = weight_variable([5, 2])
    b_fc41 = bias_variable([2])

    W_fc42 = weight_variable([5, 1])
    b_fc42 = bias_variable([1])

    return [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3, W_fc41, b_fc41, W_fc42, b_fc42]
    
    
#Definition of the __discriminator__ neural network, which
#here is responsible for detecting counterfeit (meme, label) pairs
#This whole thing should output "1" if we were dealing with real data,
#but otherwise, 0
def discrimnn(x, y, param_vec):
    """discrimnn builds the graph for a deep neural net for detecting
        counterfeit memes
        Arguments:
            x: an input tensor with the dimensions (N_examples, img_vec_size)
            y: an tensor of size (N_examples, 2) representing the class labels
                of the images represented in x
        Returns:
            a tensor of shape (N_examples, 1)
            with values equal to the probability of the input
            being a real dank/normie meme as opposed to a counterfeit
    """
    [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3, W_fc41, b_fc41, W_fc42, b_fc42] = param_vec
    #First convolutional layer, lel -- map one image to 32 base meme element maps
    #Micro-meme-magic is assumed to occur within a 5x5x3 sliding window
    with tf.name_scope('conv1'):
        h_conv1 = tf.nn.leaky_relu(conv2d_down(x, W_conv1) + b_conv1)

    #Pooling layer - downsamples by 2X
    #with tf.name_scope('pool1'):
     #   h_pool1 = avg_pool_2x2(h_conv1)

    #Meme magic mixer -- looking within the next local region (5x5), take all 32
    #feature maps, and recombine into 64 more intelligent meme-element layers
    with tf.name_scope('conv2'):
        h_conv2 = tf.nn.leaky_relu(conv2d_down(h_conv1, W_conv2) + b_conv2)
        
    #Second pooling layer, same dealio
    #with tf.name_scope('pool2'):
    #    h_pool2 = avg_pool_2x2(h_conv2)

    #Memetic mega-macro mixer. This right here is responsible for remembering
    #large-scale structure of memetic objects. Previously, we performed two
    #max pools, so our operable image size is about a fourth of its size in each dim
    #According to some meme theorists (this one), the large-scale structure
    #of a memetic object typically takes up around 1/2 of the image in each
    #dimension, and may appear anywhere within the frame.
    #To determine the output size here, we define a parameter
    #"num_big_memes", which specifies roughly how many large-scale memetic objects
    #need to be remembered to successfully classify memes.
    with tf.name_scope('conv3'):
        h_conv3 = tf.nn.leaky_relu(conv2d(h_conv2, W_conv3) + b_conv3)

    #Third pooling layer, since we only care about approximate meme positioning
    #If we started with the default width of 128, after this, we should
    #get 16x16 feature maps out of this, which ain't bad.
    with tf.name_scope('pool3'):
        h_pool3 = avg_pool_2x2(h_conv3)

    #Great, now using our incredibly high IQ, find the massive memes
    #(image-size meme macros) from all of our big memes. This is a FC layer
    with tf.name_scope('fc1'):
        h_pool3_flat = tf.reshape(h_pool3, [batch_size, l3InputSize])
        h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    #Great, now map all of the massive meme detection maps to a collection
    #of three outputs, which we will then concatenate with the actual class
    #to then determine "real/fake". Intuitively, up to this point,
    #we should have found indicators of dankness/normieness/fakeness,
    #and so in principle, the three neurons here could spike in proportion
    #to each of these, respectively. Then, for the final layer, we can
    #recombine these with what the image was labeled with to help determine
    #whether or not it really is fake.
    with tf.name_scope('fc2'):

        #short for "determined characteristics"
        det_chars = tf.nn.leaky_relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    with tf.name_scope('concat1'):
        concat_featmap = tf.concat([det_chars, y], 1)

    with tf.name_scope('fc3'):
        h_fc3 = tf.nn.leaky_relu(tf.matmul(concat_featmap, W_fc3) + b_fc3)

    with tf.name_scope('fc41'):
        logit_class = tf.matmul(h_fc3, W_fc41) + b_fc41


    with tf.name_scope('fc42'):
        logit_real = tf.matmul(h_fc3, W_fc42) + b_fc42
    
    return (logit_class, logit_real)

def deconv2d(x, W, outshape):
    """deconv2d returns a 2d deconvolution layer with full stride."""
    return tf.nn.conv2d_transpose(x, W, output_shape=outshape, strides=[1, 2, 2, 1], padding='SAME')

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_down(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
  """avg_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.contrib.layers.xavier_initializer()(shape)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


meme_base_dir = "B:/memes/resizedmemes/"

dankContent = ['rDankMemes12317/']#['CaseMemesAcademicBeans12317/', 'rBertStrips12317/', 'rBikiniBottomTwitter12317/']#,
               #'rBoneHurtingJuice12317/', 'rDankChristianMemes12317/',
               #'rDankMemes12317/', 'rDeepFriedMemes12317/', 'rPrequelMemes12317/',
               #'rSurrealMemes12317/', 'rWhoTheFuckUp12317/']
normieContent = ['rAdviceAnimals12317/'] #, 'rComedyCemetery12317/', 'rFellowKids12317/']#,
                 #'rFourPanelCringe12317/', 'rPoliticalHumor12317/', 'rTerribleFacebookMemes12317/']

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

dank_vector = np.array([0, 1])
normie_vector = np.array([1, 0])

int_to_class = [normie_vector, dank_vector]

#Loads in all of the meme data, returning two parallel lists, one
#for image data, and the other for labels, and randomizes the lists' ordering
def load_meme_data():
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

batch_size = 300

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

#Returns a (seed vector, label) pair for the generative model
def sample_gen_seed():
    labels = []
    for i in range(0, batch_size):
        labels.append(int_to_class[np.random.choice([0, 1], p=[.48, .52])])
    rand_vec = np.random.normal(0, 10, size=[batch_size, gen_seed_size])
    return (rand_vec, labels)

def flatten(x):
    shape = list(x[0].shape)
    shape[:0] = [len(x)]
    return np.concatenate(x).reshape(shape)

def output_memes_to_file(memes, labels, directory):
    for i in range(0, len(memes)):
        meme = memes[i] * 256.0
        label = labels[i]
        true_label = "normie"
        if (label[1] > 0.5):
            true_label = "dank"
        fname = true_label + "_meme_" + str(i) + ".png"
        sp.misc.imsave(directory + fname, meme)

def main(_):
  # Import data
  x_vec, y_vec = load_meme_data()

  print("Memes loaded")

  #For the purposes of the GAN here, we're just trying to generate some pretty
  #memes, so we actually don't care at all about splitting the data into folds

  #Create the random vector placeholder for the generator net
  ZX = tf.placeholder(tf.float32, shape=[None, gen_seed_size])

  #Create the random vector class label placeholder for the generator net
  ZY = tf.placeholder(tf.float32, shape=[None, 2])

  #Placeholder for real images drawn from the dataset
  X = tf.placeholder(tf.float32, shape=[None, img_width, img_height, img_color_channels])

  #Placeholder for real labels drawn from the dataset
  Y = tf.placeholder(tf.float32, shape=[None, 2])

  #Create generative and discriminative model parameters
  gen_params = gennn_params()
  discrim_params = discrimnn_params()

  G_data, G_labels = gennn(ZX, ZY, gen_params)

  #Output probabilities from the discriminator on the true data distribution
  D_class_logit_real, D_logit_real = discrimnn(X, Y, discrim_params)
  #Output probabilities from the discriminator on the counterfeit examples
  D_class_logit_fake, D_logit_fake = discrimnn(G_data, G_labels, discrim_params)

  #Add a loss term which is the similarity between the first two elements of G_data
  #G_elem1 = tf.reshape(G_data[0, :], [img_vec_size])
  #G_elem2 = tf.reshape(G_data[1, :], [img_vec_size])
  #G_elem3 = tf.reshape(G_data[2, :], [img_vec_size])


  C = (1.0 / img_vec_size) * 0.01

  def dist(x, y):
      return tf.sqrt(tf.reduce_sum(tf.square(x - y)))

  #Sum of three distances, for variety
  #sqdist = dist(G_elem1, G_elem2) + dist(G_elem2, G_elem3) + dist(G_elem1, G_elem3)
  

  #Is this loss?
  #These are the components of loss due to an incorrectly predicted source
  D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
  D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
  Loss_Source = D_loss_real + D_loss_fake

  #Class loss of D
  D_class_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_class_logit_real, labels=Y))
  D_class_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_class_logit_fake, labels=G_labels))

  Loss_Class = D_class_loss_real + D_class_loss_fake

  D_loss = Loss_Source + 3 * D_class_loss_real
  G_loss = Loss_Class - Loss_Source

  with tf.name_scope('adam_optimizer'):
    D_trainer = tf.train.AdamOptimizer(1e-2).minimize(D_loss, var_list=discrim_params)
    G_trainer = tf.train.AdamOptimizer(1e-2).minimize(G_loss, var_list=gen_params)

  #Here, we also don't care about accuracy or graphs, lel

  gc.collect()

  gen_meme_base_dir = "B:/generatedmemes/"

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_training_iters):
      x_batch, y_batch = sample_batch(x_vec, y_vec, batch_size)
      rand_vec, rand_label = sample_gen_seed()
      _, D_loss_val = sess.run([D_trainer, D_loss], feed_dict={X : x_batch, Y : y_batch, ZX : rand_vec, ZY : rand_label})
      rand_vec, rand_label = sample_gen_seed()
      _, G_loss_val = sess.run([G_trainer, G_loss], feed_dict={X : x_batch, Y : y_batch, ZX : rand_vec, ZY : rand_label})

      if i % 50 == 0:
        #Here, we'll compile the image data for a set of random init vector, label pairs,
        #and send them to a file
        rand_vec, rand_label = sample_gen_seed()
        gen_data, gen_labels = sess.run([G_data, G_labels], feed_dict={ZX : rand_vec, ZY : rand_label})
        dirname = gen_meme_base_dir + "iter" + str(i) + "/"
        if (not os.path.exists(dirname)):
            os.makedirs(dirname)
        output_memes_to_file(gen_data, gen_labels, dirname)
        print("Generated memes output to: ", dirname)
        print("D loss: ", str(D_loss_val), " G loss: ", str(G_loss_val))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/B:/memes/resizedmemes',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
        
        
