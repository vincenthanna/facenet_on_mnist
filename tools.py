import sys
import sklearn
import tensorflow as tf

import pandas as pd
import random
import numpy as np

from numpy.random import default_rng
rng = default_rng()

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import matplotlib.patheffects as PathEffects
import seaborn as sns

from sklearn.manifold import TSNE


def build_mnist_dataset(train_file_path, test_split=0.1):
    """ Transform MNIST dataset into dictionary shape.
     {number : list of images}

    Args:
        train_file_path: path to 'train.csv' file
        test_split: train_test_split ratio

    Returns:
        dictionary of train and valid
    """
    df = pd.read_csv(train_file_path)

    all_labels, all_images = sep_mnist_csv(df)

    train_images, valid_images, train_labels, valid_labels = train_test_split(all_images, all_labels, test_size=test_split)

    train_set = reorganizeMNIST(train_images, train_labels.reshape(-1))
    valid_set = reorganizeMNIST(valid_images, valid_labels.reshape(-1))

    return train_set, valid_set


def get_batch(dataset, num_samples, categorycnt, num_category_in_batch=-1):
    """Sample num_samples random images from each category of the MNIST dataset,
    returning the data along with its labels

    Args:
        dataset: dictionary of number and list of images
        num_samples: number of sample images each category(number)
        categorycnt: number of category(in MNIST, it's 10(0~9))
        num_category_in_batch : number of category in batch

    Returns:
        tuple (images, labels)
    """
    batch = []
    labels = []

    cats_in_batch = categorycnt
    if num_category_in_batch > 0:
        cats_in_batch = num_category_in_batch

    cats = [i for i in range(categorycnt)]
    if num_category_in_batch > 0:
        cats = rng.choice(categorycnt, num_category_in_batch)

    for l in cats:
        indices = rng.choice(len(dataset[l]), size=num_samples, replace=False)
        indices = np.array(indices)

        batch.append([dataset[l][i] for i in indices])
        labels += [l] * num_samples

    # image width/height
    batch = np.array(batch).reshape(cats_in_batch * num_samples, 28, 28, 1)
    labels = np.array(labels)

    # Shuffling labels and batch the same way
    s = np.arange(batch.shape[0])
    np.random.shuffle(s)

    batch = batch[s]
    labels = labels[s]

    # rescale to 0~1
    batch = batch.astype(np.float32) / 255.0

    return batch, labels


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
