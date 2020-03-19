import sys

import tensorflow as tf

import pandas as pd
import random
import numpy as np


EMBEDDING_DIM = 4 # Size of the embedding dimension (units in the last layer)
def embedImages(Images):
    conv1 = tf.layers.conv2d(Images,
                             filters=128, kernel_size=(7, 7),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer,
                             name='conv1')

    pool1 = tf.layers.max_pooling2d(conv1,
                                    pool_size=(2, 2), strides=(2, 2),
                                    padding='same',
                                    name='pool1')

    conv2 = tf.layers.conv2d(pool1,
                             filters=256, kernel_size=(5, 5),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer,
                             name='conv2')

    pool2 = tf.layers.max_pooling2d(conv2,
                                    pool_size=(2, 2), strides=(2, 2),
                                    padding='same',
                                    name='pool2')

    flat = tf.layers.flatten(pool2, name='flatten')

    # Linear activated embeddings
    embeddings = tf.layers.dense(flat,
                                 activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer,
                                 units=EMBEDDING_DIM,
                                 name='embeddings')

    embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10, name='embeddings')

    print("embeddings.shape=", embeddings.shape)

    return embeddings


def select_training_triplets_ver1(embeddings, images, labels):

    def get_dist(emb1, emb2):
        x = np.sqrt(np.subtract(emb1, emb2))
        return np.sum(x, 0)

    anchors = np.zeros(images.shape)
    positives = np.zeros(images.shape)
    negatives = np.zeros(images.shape)

    for aidx in range(len(labels)):
        current_label = labels[aidx]

        # anchor:
        anchors[aidx] = images[aidx]

        # positive:
        selected_positive = None
        ap_dist = 0
        for i in range(len(labels)):
            if i != aidx and current_label == labels[i]:
                if selected_positive is None:
                    selected_positive = images[i]
                    ap_dist = get_dist(embeddings[aidx], embeddings[i])

                # get furthest positive
                dist = get_dist(embeddings[aidx], embeddings[i])
                if dist > ap_dist:
                    ap_dist = dist
                    selected_positive = images[i]
        positives[aidx] = selected_positive

        # negative:
        selected_negative = None
        an_dist = sys.maxsize
        for i in range(len(labels)):
            if labels[aidx] != labels[i]:
                if selected_negative is None:
                    selected_negative = images[i]
                    an_dist = get_dist(embeddings[aidx], embeddings[i])

                # get closest negative but smaller than ap diff.
                dist = get_dist(embeddings[aidx], embeddings[i])
                #if dist < an_dist and dist > ap_dist:
                if dist < an_dist:
                    an_dist = dist
                    selected_negative = images[i]
        negatives[aidx] = selected_negative

    return anchors, positives, negatives



def select_training_triplets_ver2(embeddings, images, labels):

    def get_dist(emb1, emb2):
        x = np.sqrt(np.subtract(emb1, emb2))
        return np.sum(x, 0)

    ancarr = []
    posarr = []
    negarr = []

    for aidx in range(len(labels)):
        current_label = labels[aidx]

        # anchor:
        anchor = images[aidx]

        # positive:
        selected_positive = None
        ap_dist = 0
        for i in range(len(labels)):
            if i != aidx and current_label == labels[i]:
                positive = images[i]
                embedding_positive = embeddings[i]
                ap_dist = get_dist(embeddings[aidx], embeddings[i])

                # get negative
                an_dist = sys.maxsize
                negative = None
                for ni in range(len(labels)):
                    if current_label != labels[ni]:
                        if negative is None:
                            negative = images[ni]
                        dist = get_dist(embedding_positive, embeddings[ni])
                        if dist > ap_dist and dist < an_dist:
                            an_dist = dist
                            negative = images[ni]

                ancarr.append(anchor)
                posarr.append(positive)
                negarr.append(negative)

    ancarr = np.array(ancarr)
    posarr = np.array(posarr)
    negarr = np.array(negarr)

    #print(ancarr.shape, posarr.shape, negarr.shape)
    return ancarr, posarr, negarr

def triplet_loss(anchor, positive, negative):
    with tf.name_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)  # Summing over distances in each batch
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0, name='tripletloss')

    return loss

def make_loss_model(images, embed_model):
    anchors, positives, negatives = tf.split(embed_model, 3)
    print(anchors.shape)
    loss = triplet_loss(anchors, positives, negatives)
    return loss