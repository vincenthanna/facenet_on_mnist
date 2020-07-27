import sys
import sklearn
import numpy as np
import tensorflow as tf

alpha = 0.2

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
