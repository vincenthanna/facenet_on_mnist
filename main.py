import zipfile
import random

import tensorflow as tf

import pandas as pd

import numpy as np
import matplotlib.patheffects as PathEffects
import seaborn as sns

from tools import *

class FLAGS:
    def __init__(self):
        self.num_category = 10


gflags = FLAGS()


def train(df, batch_size_per_cat, num_category):

    run_test = False

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(epochs):
            print("Epoch", epoch)
            for batch in range(num_batch):

                # if batch % 10 == 0:
                #     print("Epoch", epoch, " Batch:", batch)
                images, labels = get_batch(df, batch_size_per_cat, num_category)

                feed_dict = {ph_images: images, ph_labels: labels}
                embeddings = sess.run([m_embeddings], feed_dict)
                #print("len(embeddings)=", len(embeddings))
                if type(embeddings) == list:
                    embeddings = embeddings[0]

                #print("embeddings.shape=", embeddings.shape)
                if random.randrange(0, 10) < 7:
                    select_training_triplets = select_training_triplets_ver1
                else:
                    select_training_triplets = select_training_triplets_ver2

                a,p,n = select_training_triplets(embeddings, images, labels)

                triplet_images = np.vstack([a,p,n])
                #print("triplet_images.shape", triplet_images.shape)

                feed_dict = {ph_images: triplet_images}

                _, loss_val, current_lr = sess.run([train_step, loss, optimizer._lr], feed_dict=feed_dict)
                #print("loss =", loss_val, "lr =", current_lr)


        # Training is finished, get a batch from training and validation
        # data to visualize the results
        x_train, y_train = get_batch(train_set, 32)
        x_val, y_val = get_batch(valid_set, 32)

        # Embed the images using the network
        train_embeds = sess.run(m_embeddings, feed_dict={ph_images: x_train, ph_labels:y_train})
        val_embeds = sess.run(m_embeddings, feed_dict={ph_images: x_val, ph_labels: y_val})

        if False:
            a,p,n = select_training_triplets(train_embeds, x_train, y_train)
            samplelen = 20
            fig, ax = plt.subplots(3, samplelen, figsize=(32,12))

            def __show_triplets(ax, row, imgarr, samplelen=samplelen):
                for i in range(samplelen):
                    ax[row][i].imshow(imgarr[i].reshape(28, 28), cmap=plt.cm.binary)
                    ax[row][i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

            __show_triplets(ax, 0, a)
            __show_triplets(ax, 1, p)
            __show_triplets(ax, 2, n)
            plt.show()


        tsne_train = tsne.fit_transform(train_embeds)
        tsne_val = tsne.fit_transform(val_embeds)

        scatter(tsne_train, y_train, "Results on Training Data")
        scatter(tsne_val, y_val, "Results on Validation Data")

        saver = tf.train.Saver()
        saver.save(sess, './face_model')

    return train_embeds, val_embeds


def build_facedb(dataset, num_category):
    with tf.Session() as sess:
        # load model
        saver = tf.train.import_meta_graph('face_model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))

        # Training is finished, get a batch from training and validation
        # data to visualize the results
        x, y = get_batch(dataset, 100)

        # Embed the images using the network
        embeds = sess.run(m_embeddings, feed_dict={ph_images: x, ph_labels:y})

        print(embeds.shape, y.shape)

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


def test(x, y, threshold):
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
                #print(key, dist)
                dists.append(dist)
            dists = np.array(dists)
            #print(dists.shape)
            ret = np.argmin(dists)
            retval = dists[ret]
            p = retval / np.sum(dists)
            allp = [p / np.sum(dists) for p in dists]
            #print(ret, p, y[imgidx], allp)
            #print(ret, y[imgidx])
            rets.append(ret)

        rets = np.array(rets)
        print(rets.shape, y.shape)
        acc = ((rets == y).mean())
        print("valid accuracy : ", acc)

def run():
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context('notebook', font_scale=1.5, rc={"lines.linewidth": 2.5})

    zp = zipfile.ZipFile("mnist.zip")
    zp.extractall('./')

    df_train = pd.read_csv("./train.csv")
    df_test = pd.read_csv("./test.csv")

    all_labels, all_images = sep_mnist_csv(df_train)

    train_images, valid_images, train_labels, valid_labels = train_test_split(all_images, all_labels, test_size=0.20)

    print(train_images.shape, train_labels.shape)
    print(valid_images.shape, valid_labels.shape)

    num_category = gflags.num_category = 10
    batch_size_per_cat = gflags.batch_size_per_cat = 10

    train_set = reorganizeMNIST(train_images, train_labels.reshape(-1))
    valid_set = reorganizeMNIST(valid_images, valid_labels.reshape(-1))

    test_batch(train_set, num_category)

    gflags.num_batch = int(len(train_set[0]) / batch_size_per_cat)
    gflags.batch_size = batch_size_per_cat * num_category
    print("num_batch =", gflags.num_batch, " batch_size=", gflags.batch_size)


if __name__ == '__main__':
    run()
