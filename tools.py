import sys
import sklearn
import tensorflow as tf

import pandas as pd
import random
import numpy as np

from np.random import default_rng
rng = default_rng()

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import matplotlib.patheffects as PathEffects
import seaborn as sns

from sklearn.manifold import TSNE

num_category = 10

def get_batch(dataset, num_samples, categorycnt=num_category):
    """Sample num_samples random images from each category of the MNIST dataset,
    returning the data along with its labels

    Args:
        dataset: dictionary of number and list of images
        num_samples: number of sample images each category(number)
        categorycnt: number of category(in MNIST, it's 10(0~9))

    Returns:
        tuple (images, labels)
    """
    batch = []
    labels = []

    for l in range(categorycnt):
        # indices = np.random.randint(0, len(dataset[l]), num_samples)
        indices = rng.choice(len(dataset[l]), size=num_samples, replace=False)
        indices = np.array(indices)

        batch.append([dataset[l][i] for i in indices])
        labels += [l] * num_samples

    # image width/height
    batch = np.array(batch).reshape(categorycnt * num_samples, 28, 28, 1)
    labels = np.array(labels)

    # Shuffling labels and batch the same way
    s = np.arange(batch.shape[0])
    np.random.shuffle(s)

    batch = batch[s]
    labels = labels[s]

    # rescale to 0~1
    batch = batch.astype(np.float32) / 255.0

    return batch, labels


# def test_batch():
#     """check get_batch function works correctly"""
#     images, labels = get_batch(train_set, 20, num_category)
#     fig, ax = plt.subplots(2, 10, figsize=(24, 8))
#     for i in range(2):
#         for j in range(10):
#             ax[i][j].imshow(images[2 * i + j].reshape(28, 28), cmap=plt.cm.binary)
#             ax[i][j].set_title(str(labels[2 * i + j]))
#             ax[i][j].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,
#                                  left=False, labelleft=False)
#     plt.show()


def sep_mnist_csv(df):
    """ separate mnist DataFrame into image and labels

    Args:
        df: MNIST dataset ['label', 'image']

    Return:
        tuple of numpy array (labels, images)
    """

    df_labels = df['label'].values
    df_images = df.drop(columns=['label']).values

    return df_labels, df_images


def reorganizeMNIST(x, y):
    """Reconstruct data into dictionary container of number and list of images

    Args:
        x: numpy array of images
        y: labels

    Return:
        dictionary consists of label(number) and array of images
    """

    assert x.shape[0] == y.shape[0]

    dataset = {i: [] for i in range(10)}

    for i in range(x.shape[0]):
        dataset[y[i]].append(x[i])

    return dataset


def scatter(x, labels, subtitle=None):
    """draw scatter plot

    Args:
        x: 2D array of shape (?, 2) ([?, 0] as x value and [?, 1] as y value)
        labels: labels of x
        subtitle: subtitle of plot

    Return:
        None
    """

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context('notebook', font_scale=1.5, rc={"lines.linewidth": 2.5})

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])

    if subtitle != None:
        plt.suptitle(subtitle)

    plt.show()


def select_training_triplets(embeddings, images, labels):
    """make triplet arrays from input

    Args:
        embeddings : embeddings of images 'images'
        images : numpy array of images
        labels : labels array

    Return:
        Arrays of triplet elements
            array of anchor images
            array of selected positive images
            array of selected negative images
    """

    def get_dist(emb1, emb2):
        x = np.sqrt(np.square(np.subtract(emb1, emb2)))
        return np.sum(x, 0)

    ancarr = []
    posarr = []
    negarr = []

    do_shuffle = True

    for aidx in range(len(labels)):
        current_label = labels[aidx]

        # anchor: current image
        anchor = images[aidx]

        # positive:
        selected_positive = None
        ap_dist = 0
        if do_shuffle:
            pos_indices = sklearn.utils.shuffle([i for i in range(len(labels))])
        else:
            pos_indices = [i for i in range(len(labels))]
        for i in pos_indices:
            if i != aidx and current_label == labels[i]:
                # consider all other images in the same category as positive.
                positive = images[i]
                embedding_positive = embeddings[i]
                ap_dist = get_dist(embeddings[aidx], embeddings[i])
                # if ap_dist == 0.0:
                #     print("ap_dist is 0 :", aidx, embeddings[aidx], i, embeddings[i])

                # get negative(semi-hard)
                an_dist = sys.maxsize
                negative = None

                neg_indices = [i for i in range(len(labels))]
                if do_shuffle:
                    neg_indices = sklearn.utils.shuffle([i for i in range(len(labels))])
                for ni in neg_indices:
                    if current_label != labels[ni]:
                        if negative is None:
                            negative = images[ni]
                            an_dist = get_dist(embedding_positive, embeddings[ni])
                        dist = get_dist(embedding_positive, embeddings[ni])
                        """select negative that meet conditions below
                            - dist(a, n) is smallest
                            - dist(a, n) > dist(a, p)
                        """

                        # if dist > ap_dist and dist < an_dist:
                        if dist > ap_dist:
                            an_dist = dist
                            negative = images[ni]

                if ap_dist < an_dist:
                    # print("ap_dist=", ap_dist, "an_dist=", an_dist)
                    ancarr.append(anchor)
                    posarr.append(positive)
                    negarr.append(negative)
    if do_shuffle:
        ancarr, posarr, negarr = sklearn.utils.shuffle(ancarr, posarr, negarr)

    # print("selected triples : ", len(ancarr))

    ancarr = np.array(ancarr)
    posarr = np.array(posarr)
    negarr = np.array(negarr)

    return ancarr, posarr, negarr
