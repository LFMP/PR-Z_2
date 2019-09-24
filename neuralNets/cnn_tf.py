import tensorflow as tf
import pandas as pd
import numpy as np

# --------------------------------------------------------- HELPER FUNCTIONS

# INIT WEIGHTS
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init_random_dist)

# INIT BIAS
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

# CONV2D
# x (input tensor) --> [batch, Height, Weight, Channels]
# W (kernel) --> [filter Height, filter Weight, Channels in, Channels out]
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# POOLING
# x --> [batch, h, w, c]
def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# CONVoLUTIONAL LAYER
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

# Normal (Fully connected)
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

# ---------------------------------------------------------------- CONSTANTS
IMG_HEIGHT = 103
IMG_WIDTH = 134
NUM_COLOR_CHANNELS = 3
NUM_CLASSES = 2
LEARNING_RATE = 0.0001
STEPS = 500
# ----------------------------------------------------------------- LOAD DATASET

# ----------------------------------------------------------------- MODEL
x_data = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, NUM_COLOR_CHANNELS])
y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

# probability for the drop out
hold_prob = tf.placeholder(tf.float32)

# Layers
# created according to the 'Roecker et al. (2018)' model
layer_1 = convolutional_layer(x_data, [3, 3, NUM_COLOR_CHANNELS, 32])
layer_2 = convolutional_layer(layer_1, [3, 3, 32, 32])
layer_3 = max_pool_2by2(layer_2)

layer_4 = convolutional_layer(layer_3, [3, 3, 32, 32])
layer_5 = convolutional_layer(layer_4, [3, 3, 32, 32])
layer_6 = max_pool_2by2(layer_5)

flatten = tf.reshape(layer_6, [-1, 8*8*64])

layer_7 = tf.nn.relu(normal_full_layer(flatten, 512))
layer_8 = tf.nn.relu(normal_full_layer(layer_7, 512))

# Dropout
dropout = tf.nn.dropout(layer_8, keep_prob=hold_prob)

y_pred = normal_full_layer(dropout, NUM_CLASSES)

# LOSS FUNCTION
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))

# OPTIMIZER
# TODO finish this file
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
