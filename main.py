import zipfile
import random

import tensorflow as tf
import pandas as pd

import numpy as np
import matplotlib.patheffects as PathEffects
import seaborn as sns

from model import nn2
import argparse

from tools import *
from model.nn2 import *
from model.triplet import *

from sklearn.manifold import TSNE

tsne = TSNE()

tf.reset_default_graph()

class FLAGS:
    def __init__(self):
        self.num_category = 10


logdir="logs/fit/"


gflags = FLAGS()


# make dataset
#############################################################



#############################################################



'''
# Placeholders for inserting data
ph_images = tf.placeholder(tf.float32, [None, 28, 28, 1], name='images_ph')
ph_labels = tf.placeholder(tf.int32, [None], name='labels_ph')
lr = tf.Variable(0.001)
alpha = tf.Variable(0.2)

# make models
m_embeddings = nn2(ph_images, 3) # embedding model for test/use
loss = make_loss_model(ph_images, m_embeddings)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_step = optimizer.minimize(loss=loss)
'''



epochs = 1

def train(train_dataset, batch_size_per_cat, num_category, epochs, num_batch, learning_rate, \
          ph_images, ph_labels, m_embeddings, m_loss, m_train, optimizer, logdir="logs/fit"):

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        summary_writer = tf.summary.FileWriter(logdir)
        summary_writer.add_graph(sess.graph)
        writer_op = tf.summary.merge_all()

        for epoch in range(epochs):

            for batch in range(num_batch):

                images, labels = get_batch(train_dataset, batch_size_per_cat, num_category)

                feed_dict = {ph_images: images, ph_labels: labels}
                embeddings = sess.run([m_embeddings], feed_dict)
                if type(embeddings) == list:
                    embeddings = embeddings[0]

                a, p, n = select_training_triplets(embeddings, images, labels)

                if len(a) > 0:
                    triplet_images = np.vstack([a, p, n])
                    # print("triplet_images.shape", triplet_images.shape)

                    feed_dict = {ph_images: triplet_images}

                    # print("sess.run() : ", type(train_step), type(loss), type(optimizer._lr), type(writer_op) )
                    _, loss_val, current_lr = sess.run([m_train, m_loss, optimizer._lr], feed_dict=feed_dict)
                    # summary_writer.add_summary(summary, epoch * num_batch + batch)
                    # print("loss =", loss_val, "lr =", current_lr)

        saver = tf.train.Saver()
        saver.save(sess, './face_model')

        # Training is finished, get a batch from training and validation
        # data to visualize the results
        x_train, y_train = get_batch(train_dataset, 32)

        # Embed the images using the network
        embeds = sess.run(m_embeddings, feed_dict={ph_images: x_train, ph_labels: y_train})
        transformed_output = tsne.fit_transform(embeds)
        scatter(transformed_output, y_train, "Results on Training Data")
    return embeds


def build_facedb(dataset, ph_images, ph_labels, m_embedding, num_category):
    sample_cnt_by_category = 100
    with tf.Session() as sess:
        # load model
        saver = tf.train.import_meta_graph('face_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        # Training is finished, get a batch from training and validation
        # data to visualize the results
        x, y = get_batch(dataset, sample_cnt_by_category, num_category)

        # Embed the images using the network
        embeds = sess.run(m_embedding, feed_dict={ph_images: x, ph_labels:y})

        embed_dict = {new_list : [] for new_list in range(num_category)}

        for i, v in enumerate(y):
            embed_dict[v].append(embeds[i])

        print(embed_dict[0])

        facedb = {}
        for key in range(num_category):
            vals = [img for i, img in enumerate(embeds) if y[i] == key]
            vals = np.array(vals).mean(axis=0)
            facedb[key] = vals

        return facedb


def test(x, y, threshold, facedb, ph_images, m_embeddings):
    ph_images = tf.placeholder(tf.float32, [None, 28, 28, 1], name='images_ph')
    with tf.Session() as sess:

        # load model
        saver = tf.train.import_meta_graph('face_model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))

        rets = []
        for imgidx in range(y.shape[0]):
            face_img = x[imgidx].reshape([1, *x[imgidx].shape])

            encoding = sess.run(m_embeddings, feed_dict={ph_images: face_img})
            encoding.reshape([-1])

            dists = []
            for key in facedb:
                dist = np.linalg.norm(encoding - facedb[key])                
                dists.append(dist)
            dists = np.array(dists)
            ret = np.argmin(dists)
            retval = dists[ret]
            p = retval / np.sum(dists)
            allp = [p / np.sum(dists) for p in dists]
            rets.append(ret)

        rets = np.array(rets)
        print(rets.shape, y.shape)
        acc = ((rets == y).mean())
        print("Valid accuracy : ", acc)

        inaccurace_indices = np.nonzero(rets != y)
        inacc_cnt = inaccurace_indices[0].shape[0]

        print("")
        print("correct/incorrect predicted images:")
        accurace_indices = np.nonzero(rets == y)
        acc_cnt = accurace_indices[0].shape[0]
        show_max = 10
        fig, ax = plt.subplots(2, show_max, figsize=(2 * show_max, 4))
        # show correctly predicted images
        for i, idx in enumerate(accurace_indices[0]):
            ax[0][i].imshow(x[idx].reshape(28, 28), cmap=plt.cm.binary)
            ax[0][i].set_title(str(rets[idx]))
            ax[0][i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,
                                 left=False, labelleft=False)
            if i >= show_max - 1:
                break

        # show mispredicted images
        for i, idx in enumerate(inaccurace_indices[0]):
            ax[1][i].imshow(x[idx].reshape(28, 28), cmap=plt.cm.binary)
            ax[1][i].set_title(str(rets[idx]))
            ax[1][i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,
                                 left=False, labelleft=False)
            if i >= show_max - 1:
                break

        plt.show()



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('lr', type=float,
                        help='Learning rate for training.', default=0.001)
    parser.add_argument('epochs', type=int,
                        help='Number of epochs on training', default=2)
    parser.add_argument("batch_per_category", type=int,
                        help='Number of data per category within single batch', default=4)

    return parser.parse_args(argv)


def run(args):
    global gflags

    print(args)

    epochs = args.epochs
    batch_size_per_cat = args.batch_per_category
    lr = args.lr
    print(args.epochs, args.lr, args.batch_per_category)

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context('notebook', font_scale=1.5, rc={"lines.linewidth": 2.5})

    zp = zipfile.ZipFile("mnist.zip")
    zp.extractall('./')

    train_set, valid_set = build_mnist_dataset("./mnist_train.csv")

    num_batch = gflags.num_batch = int(len(train_set[0]) / batch_size_per_cat)
    gflags.batch_size = batch_size_per_cat * num_category
    print("num_batch =", gflags.num_batch, " batch_size=", gflags.batch_size)

    # variables and parameters
    learning_rate = tf.Variable(lr)
    ph_images = tf.placeholder(tf.float32, [None, 28, 28, 1], name='images_ph')
    ph_labels = tf.placeholder(tf.int32, [None], name='labels_ph')

    # models for embedding, training
    embedding_model = nn2(ph_images, 3)  # embedding model for test/use
    loss = make_loss_model(ph_images, embedding_model)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss=loss)

    
    # Turn on if you want to check code is valid(quick one-time run)
    if True:
        epochs = 1
        num_batch = gflags.num_batch = 2

    # train model and make embeddings with it
    train(df=train_set, batch_size_per_cat=batch_size_per_cat, num_category=num_category, epochs=epochs, num_batch=num_batch)

    # make embeddings for each labels
    facedb = build_facedb(train_set, embedding_model, num_category)
    print(facedb)

    # test with valid set.
    x_val, y_val = get_batch(valid_set, 32, gflags.num_category)
    test(x_val, y_val, threshold=0.1, facedb=facedb)

if __name__ == '__main__':
    run(parse_arguments(sys.argv[1:]))

