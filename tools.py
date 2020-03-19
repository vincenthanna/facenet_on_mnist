import tensorflow as tf

import pandas as pd
import random
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import matplotlib.patheffects as PathEffects
import seaborn as sns

from sklearn.manifold import TSNE


def get_batch(dataset, num_samples, categorycnt):
    # Sample BATCH_K random images from each category of the MNIST dataset,
    # returning the data along with its labels
    batch = []
    labels = []

    for l in range(categorycnt):
        indices = np.random.randint(0, len(dataset[l]), num_samples)
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

    return batch, labels

def test_batch(df, num_category):
    images, labels = get_batch(df, 20, num_category)
    fig, ax = plt.subplots(2, 10, figsize=(24,8))
    for i in range(2):
        for j in range(10):
            ax[i][j].imshow(images[2 * i + j].reshape(28, 28), cmap=plt.cm.binary)
            ax[i][j].set_title(str(labels[2 * i + j]))
            ax[i][j].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.show()

def sep_mnist_csv(df):
    """ separate mnist DataFrame into image and labels"""

    df_labels = df['label'].values
    df_images = df.drop(columns=['label']).values

    return df_labels, df_images

def reorganizeMNIST(x, y):
    """Reconstruct data into dictionary container of number and list of images"""
    assert x.shape[0] == y.shape[0]

    dataset = {i: [] for i in range(10)}

    for i in range(x.shape[0]):
        dataset[y[i]].append(x[i])

    return dataset


def scatter(x, labels, subtitle=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    if subtitle != None:
        plt.suptitle(subtitle)

    plt.show()


