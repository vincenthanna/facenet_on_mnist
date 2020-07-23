import sys
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from inspect import currentframe

tf.reset_default_graph()

model_debug = True


def resize(input, size):
    """add resize-image layer to input

    Args:
        input: input layer
        size: size by tuple (width, height)

    Return:
        layer with resize_images added.
    """
    layer = tf.image.resize_images(input, size, method=tf.image.ResizeMethod.BICUBIC, align_corners=True,
                                   preserve_aspect_ratio=True)
    return layer


def conv(input, inch, outch, filter_h, filter_w, stride_h, stride_w, padding='SAME', name='conv_layer'):
    """make conv2d layer with parameters

    Args:
        input: input layer
        inch: number of input channel
        outch: number of output channel
        filter_h: filter height
        filter_w: filter widht
        stride_h: stride height
        stride_w: stride width
        padding: padding option passing to conv2d()
        name: name scope string

    Return:
        layer with conv2d added.
    """
    with tf.name_scope(name) as scope:
        layer = tf.layers.conv2d(input, outch, filter_h, strides=(stride_h, stride_w), padding="same",
                                 activation=tf.nn.relu)
        return layer


def maxpool(input, filter_h, filter_w, stride_h, stride_w, padding, name):
    """add max pooling layer to input

    Args:
        input: input layer
        filter_h: filter height
        filter_w: filter widht
        stride_h: stride height
        stride_w: stride width
        padding: padding option passing to conv2d()
        name: name scope string

    Return:
        layer with max pooling added.
    """
    with tf.name_scope(name):
        mp = tf.nn.max_pool(input, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1],
                            padding=padding)
        # print(name + " : ", str(mp.shape))
        return mp


def avgpool(input, filter_h, filter_w, stride_h, stride_w, padding, name):
    """add average pooling layer to input

    Args:
        input: input layer
        filter_h: filter height
        filter_w: filter widht
        stride_h: stride height
        stride_w: stride width
        padding: padding option passing to conv2d()
        name: name scope string

    Return:
        layer with average pooling added.
    """
    with tf.name_scope(name):
        ret = tf.nn.avg_pool(input, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1],
                             padding=padding)
        print(name, " : ", str(ret.shape))
        return ret


def l2pool(input, filter_h, filter_w, stride_h, stride_w, padding, name):
    """add L2 pooling layer to input
    L2-norm pooling selects average of squared rectangular neighborhood & apply square root.

    Args:
        input: input layer
        filter_h: filter height
        filter_w: filter widht
        stride_h: stride height
        stride_w: stride width
        padding: padding option passing to conv2d()
        name: name scope string

    Return:
        layer with average pooling added.
    """
    with tf.name_scope(name):
        squared = tf.square(input)
        subsample = tf.nn.avg_pool(squared, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1],
                                   padding=padding)
        subsample_sum = tf.multiply(subsample, filter_h * filter_w)
        return tf.sqrt(subsample_sum)


def flatten(input, name):
    """convert CNN layer to fully-connected layer

    Args:
        input: input layer
        name: name scope string

    Return:
        layer with flatten layer is attached.
    """
    with tf.name_scope(name):
        l = tf.layers.flatten(input)
    return l


def inception(input_layer, input_size, conv_stride_hw, o1x1, o3x3r, o3x3, o5x5r, o5x5, pooling, pool_filter_size,
              pool_stride_hw, pool_reduce_outch, name, useOnlyMaxPool=True):
    """Make GoogLeNet Inception v1 layer

    Args:
        input_layer: input layer
        input_size: number of channels of input layer
        conv_stride_hw: tuple of stride height & width (h, w)
        o1x1: 1x1 convolution out channel count
        o3x3r: number of out channel of 1x1 convolution layer ahead of 3x3 convolution layer
        o3x3: number of out channel of 3x3 convolution layer
        o5x5r: number of out channel of 1x1 convolution layer ahead of 5x5 convolution layer
        o5x5: number of out channel of 5x5 convolution layer
        pooling: pooling method of "3x3 max pooling"
            Max pooling or L2-norm pooling can be used in "3x3 max pooling" layer
        pool_filter_size: height & width of filter size of pooling layer
        pool_stride_hw: stride value of 3x3 max pooling
        pool_reduce_outch: if not zero, number of out channel of 1x1 convolution ahead of 3x3 max pooling
        name: name scope string
        useOnlyMaxPool: use only max pooling(Not using L2-norm pooling)

    Return:
        layer with inception layer attached.

    """
    global model_debug
    debug = model_debug
    if False:
        print('name = ', name)
        print('inputSize = ', input_size)
        print('kernelSize = {3,5}')
        print('outputSize = {%d,%d}' % (o3x3, o5x5))
        print('reduceSize = {%d,%d,%d,%d}' % (o3x3r, o5x5r, pool_reduce_outch, o1x1))
        print('pooling = {%s, %d, %d, %d, %d}' % (
        pooling, pool_filter_size, pool_filter_size, pool_stride_hw[0], pool_stride_hw[1]))

    if (pool_reduce_outch > 0):
        pool_out_cnt = pool_reduce_outch
    else:
        pool_out_cnt = input_size

    outputs = []

    with tf.name_scope(name):
        if o1x1 > 0:
            l = conv(input_layer, input_size, o1x1, 1, 1, 1, 1, padding='SAME', name='in1_conv1x1')
            outputs.append(l)

        if o3x3 > 0:
            l = conv(input_layer, input_size, o3x3r, 1, 1, 1, 1, padding='SAME', name='in2_conv1x1')
            l = conv(l, o3x3r, o3x3, 3, 3, conv_stride_hw[0], conv_stride_hw[1], padding='SAME', name='in2_conv3x3')
            outputs.append(l)

        if o5x5 > 0:
            l = conv(input_layer, input_size, o5x5r, 1, 1, 1, 1, padding='SAME', name='in3_conv1x1')
            l = conv(l, o5x5r, o5x5, 5, 5, conv_stride_hw[0], conv_stride_hw[1], padding='SAME', name='in3_conv5x5')
            outputs.append(l)

        if pooling != None:
            # Either max pooling or l2 pooling is used in pooling layer.
            '''
            if pooling == 'max':
                pool = maxpool(input_layer, pool_filter_size, pool_filter_size, pool_stride_hw[0], pool_stride_hw[1], padding='SAME', name='maxpool')
            else:
                pool = l2pool(input_layer, pool_filter_size, pool_filter_size, pool_stride_hw[0], pool_stride_hw[1], padding='SAME', name='l2pool')
            '''
            pool = maxpool(input_layer, pool_filter_size, pool_filter_size, pool_stride_hw[0], pool_stride_hw[1],
                           padding='SAME', name='maxpool')

            if pool_reduce_outch > 0:
                poolconv = conv(pool, input_size, pool_reduce_outch, 1, 1, 1, 1, padding='SAME', name='in4_conv1x1')
            else:
                poolconv = pool

            outputs.append(poolconv)

        # if debug:
        #     print("outputs : " + str([str(o.shape) for o in outputs]))
        l = array_ops.concat(outputs, axis=3, name=name)
        if debug:
            print(name + " output : " + str(l.shape))
        return l


def nn2(input, inch):
    """Make NN2 model

    Args:
        input: input layer
        inch: number of channels in input layer.

    Return:
        nn2 model
    """

    global model_debug
    debug = model_debug
    print("nn2 input:", input.shape)
    l = input

    ################################################
    # temp code for MNIST dataset resize
    ################################################
    l = tf.image.grayscale_to_rgb(l)
    l = resize(l, (224, 224))  # resize
    l = tf.cast(l > 0.5, dtype=tf.float32)
    ################################################

    l = conv(l, inch, 64, 7, 7, 2, 2, padding='SAME', name='nn_conv1')
    if debug:
        print('conv1' + " output : ", str(l.shape))
    l = maxpool(l, 3, 3, 2, 2, padding='SAME', name='nn_maxpool1')
    if debug:
        print('maxpool1' + " output : ", str(l.shape))

    l = inception(l, 64, (1, 1), 0, 64, 192, 0, 0, None, 0, (0, 0), 0, name="nn_convolution_2")
    l = maxpool(l, 3, 3, 2, 2, 'SAME', "nn_maxpool2")
    if debug:
        print('maxpool2' + " output : ", str(l.shape))
    l = inception(l, 256, (1, 1), 64, 96, 128, 16, 32, 'max', 3, (1, 1), 32, name="inception_3a")
    l = inception(l, 320, (1, 1), 64, 96, 128, 32, 64, 'l2', 3, (1, 1), 64, name="inception_3b")
    l = inception(l, 640, (2, 2), 0, 128, 256, 32, 64, 'max', 3, (2, 2), 0, name="inception_3c")

    l = inception(l, 640, (1, 1), 256, 96, 192, 32, 64, 'l2', 3, (1, 1), 128, name="inception_4a")
    l = inception(l, 640, (1, 1), 224, 112, 224, 32, 64, 'l2', 3, (1, 1), 128, name="inception_4b")
    l = inception(l, 640, (1, 1), 192, 128, 256, 32, 64, 'l2', 3, (1, 1), 128, name="inception_4c")
    l = inception(l, 640, (1, 1), 160, 144, 288, 32, 64, 'l2', 3, (1, 1), 128, name="inception_4d")
    l = inception(l, 640, (2, 2), 0, 160, 256, 64, 128, 'l2', 3, (2, 2), 0, name="inception_4e")

    l = inception(l, 1024, (1, 1), 384, 192, 384, 48, 128, 'l2', 3, (1, 1), 128, name="inception_5a")
    l = inception(l, 1024, (1, 1), 384, 192, 384, 48, 128, 'max', 3, (1, 1), 128, name="inception_5b")

    l = avgpool(l, 7, 7, 1, 1, 'VALID', 'nn_avg_pooling')

    l = flatten(l, "nn_to_fc")
    if debug:
        print("flatten" + " output : ", str(l.shape))

    l = tf.layers.dense(l, 128, activation=None)

    ################################################
    # temp code for MNIST
    ################################################
    l = tf.layers.dense(l, 4, activation=None)
    ################################################

    l = tf.math.l2_normalize(l, axis=1, epsilon=1e-10, name='final_embeddings')

    print("model final output : ", str(l.shape))

    return l

# img = tf.placeholder(tf.float32, [None, 28, 28, 1], name='img')
# ret = nn2(img, 3)
# print(ret.shape)
