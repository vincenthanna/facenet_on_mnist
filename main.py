import zipfile
import random

import tensorflow as tf

import pandas as pd

import numpy as np
import matplotlib.patheffects as PathEffects
import seaborn as sns

from tools import *
from model.simple_nn import *

from sklearn.manifold import TSNE

tf.reset_default_graph()

class FLAGS:
    def __init__(self):
        self.num_category = 10


gflags = FLAGS()

# Placeholders for inserting data
ph_images = tf.placeholder(tf.float32, [None, 28, 28, 1], name='images_ph')
ph_labels = tf.placeholder(tf.int32, [None], name='labels_ph')
lr = tf.Variable(0.001)
alpha = tf.Variable(0.2)

# make models
m_embeddings = embedImages(ph_images)
loss = make_loss_model(ph_images, m_embeddings, alpha)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_step = optimizer.minimize(loss=loss)

def train(df, batch_size_per_cat, num_category, epochs, num_batch):
    """train model

    Args:

    Returns:

    Raises:
        None
    """    

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(epochs):
            print("Epoch", epoch)
            for batch in range(num_batch):

                images, labels = get_batch(df, batch_size_per_cat, num_category)

                feed_dict = {ph_images: images, ph_labels: labels}
                embeddings = sess.run([m_embeddings], feed_dict)
                #print("len(embeddings)=", len(embeddings))
                if type(embeddings) == list:
                    embeddings = embeddings[0]

                select_training_triplets = select_training_triplets_ver2                    

                a,p,n = select_training_triplets(embeddings, images, labels)

                triplet_images = np.vstack([a,p,n])
                #print("triplet_images.shape", triplet_images.shape)

                feed_dict = {ph_images: triplet_images}

                _, loss_val, current_lr = sess.run([train_step, loss, optimizer._lr], feed_dict=feed_dict)
                #print("loss =", loss_val, "lr =", current_lr)

        # Training is finished, get a batch from training and validation
        # data to visualize the results
        x, y = get_batch(df, 32, num_category)

        # Embed the images using the network
        embeds = sess.run(m_embeddings, feed_dict={ph_images: x, ph_labels:y})        

        if False:
            a,p,n = select_training_triplets(embeds, x, y)
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

            tsne = TSNE()
            tsne_train = tsne.fit_transform(embeds)
            scatter(tsne_train, y, "Results on Data")            

        # save model as file
        saver = tf.train.Saver()
        saver.save(sess, './face_model')

    return None


def build_facedb(dataset, model_embedding, num_category):
    global gflags
    sample_cnt_by_category = 100
    with tf.Session() as sess:
        # load model
        saver = tf.train.import_meta_graph('face_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        # Training is finished, get a batch from training and validation
        # data to visualize the results
        x, y = get_batch(dataset, sample_cnt_by_category, gflags.num_category)

        # Embed the images using the network
        embeds = sess.run(model_embedding, feed_dict={ph_images: x, ph_labels:y})        

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


def test(x, y, threshold, facedb):
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


        # show mis-predicted images
        print("")
        print("wrong predicted images:")
        inaccurace_indices = np.nonzero(rets != y)
        inacc_cnt = inaccurace_indices[0].shape[0]
        inacc_cnt = min(inacc_cnt, 20)
        fig, ax = plt.subplots(1, inacc_cnt, figsize=(2 * inacc_cnt, 4)) 
        for i, idx in enumerate(inaccurace_indices[0]):
            if i < inacc_cnt:
                ax[i].imshow(x[idx].reshape(28, 28), cmap=plt.cm.binary)
                ax[i].set_title(str(rets[idx]))
                ax[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            else:
                break

        plt.show()

def run():
    global gflags

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
    batch_size_per_cat = gflags.batch_size_per_cat = 4

    train_set = reorganizeMNIST(train_images, train_labels.reshape(-1))
    valid_set = reorganizeMNIST(valid_images, valid_labels.reshape(-1))

    #test_batch(train_set, num_category)

    num_batch = gflags.num_batch = int(len(train_set[0]) / batch_size_per_cat)
    gflags.batch_size = batch_size_per_cat * num_category 
    print("num_batch =", gflags.num_batch, " batch_size=", gflags.batch_size)

    epochs = 10
    
    # Turn on if you want to check code is valid(quick one-time run)
    if False:
        epochs = 1
        num_batch = gflags.num_batch = 2

    # train model and make embeddings with it
    train(df=train_set, batch_size_per_cat=batch_size_per_cat, num_category=num_category, epochs=epochs, num_batch=num_batch)

    # make embeddings for each labels
    facedb = build_facedb(train_set, m_embeddings, num_category)
    print(facedb)

    # test with valid set.
    x_val, y_val = get_batch(valid_set, 32, gflags.num_category)
    test(x_val, y_val, threshold=0.1, facedb=facedb)    

if __name__ == '__main__':
    run()

