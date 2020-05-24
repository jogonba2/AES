import tensorflow as tf
from tensorflow.keras import backend as K


# Improve this! #
def pairwise_cosine_similarity_rank(candidates, positives):
    ranking = {}

    candidates = tf.math.l2_normalize(candidates, axis=-1)
    positives = tf.math.l2_normalize(positives, axis=-1)

    for i in range(candidates.shape[0]):
        ranking[i] = (None, float("-inf"))
        candidate = candidates[i]
        for j in range(positives.shape[0]):
            if i==j:
                continue
            positive = positives[j]
            sim = -tf.keras.losses.cosine_similarity(candidate, positive).numpy()
            if sim > ranking[i][1]:
                ranking[i] = (j, sim)

    return ranking

# Improve this! #
def pairwise_euclidean_distance_rank(candidates, positives):
    ranking = {}

    candidates = tf.math.l2_normalize(candidates, axis=-1)
    positives = tf.math.l2_normalize(positives, axis=-1)

    for i in range(candidates.shape[0]):
        ranking[i] = (None, float("inf"))
        candidate = candidates[i]
        for j in range(positives.shape[0]):
            if i==j:
                continue
            positive = positives[j]
            dist = K.sqrt(K.sum(K.square(candidate - positive), axis=-1)).numpy()
            if dist < ranking[i][1]:
                ranking[i] = (j, dist)

    return ranking


def euclidean_distance_pos_pair(_, y_pred):
    vector_size = y_pred.shape[-1] // 3
    anchor = y_pred[:, :vector_size]
    positive = y_pred[:, vector_size:2 * vector_size]
    anchor = tf.math.l2_normalize(anchor)
    positive = tf.math.l2_normalize(positive)
    return K.sqrt(K.sum(K.square(anchor - positive), axis=-1))


def euclidean_distance_neg_pair(_, y_pred):
    vector_size = y_pred.shape[-1] // 3
    anchor = y_pred[:, :vector_size]
    negative = y_pred[:, 2 * vector_size:]
    anchor = tf.math.l2_normalize(anchor)
    negative = tf.math.l2_normalize(negative)
    return K.sqrt(K.sum(K.square(anchor - negative), axis=-1))


def cosine_similarity_pos_pair(_, y_pred):
    vector_size = y_pred.shape[-1] // 3
    anchor = y_pred[:, :vector_size]
    positive = y_pred[:, vector_size:2 * vector_size]
    anchor = tf.math.l2_normalize(anchor)
    positive = tf.math.l2_normalize(positive)
    return -tf.keras.losses.cosine_similarity(anchor, positive, axis=-1)


def cosine_similarity_neg_pair(_, y_pred):
    vector_size = y_pred.shape[-1] // 3
    anchor = y_pred[:, :vector_size]
    negative = y_pred[:, 2 * vector_size:]
    anchor = tf.math.l2_normalize(anchor)
    negative = tf.math.l2_normalize(negative)
    return -tf.keras.losses.cosine_similarity(anchor, negative, axis=-1)