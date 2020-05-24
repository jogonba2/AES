from tensorflow.keras import backend as K
import tensorflow as tf


def euclidean_triplet_loss(margin=1):
    def f(_, y_pred):
        vector_size = y_pred.shape[-1] // 3
        anchor = y_pred[:, :vector_size]
        positive = y_pred[:, vector_size:2*vector_size]
        negative = y_pred[:, 2*vector_size:]

        # Normalize #
        anchor = tf.math.l2_normalize(anchor)
        positive = tf.math.l2_normalize(positive)
        negative = tf.math.l2_normalize(negative)

        # Compute distances #
        pos_dist = K.sqrt(K.sum(K.square(anchor - positive), axis=-1))
        neg_dist = K.sqrt(K.sum(K.square(anchor - negative), axis=-1))

        return K.maximum(pos_dist - neg_dist + margin, 0)

    return f

def cosine_triplet_loss(margin=1):
    def f(_, y_pred):
        vector_size = y_pred.shape[-1] // 3
        anchor = y_pred[:, :vector_size]
        positive = y_pred[:, vector_size:2*vector_size]
        negative = y_pred[:, 2*vector_size:]

        # Normalize #
        anchor = tf.math.l2_normalize(anchor)
        positive = tf.math.l2_normalize(positive)
        negative = tf.math.l2_normalize(negative)

        # Compute similarities #
        pos_sim = -tf.keras.losses.cosine_similarity(anchor, positive, axis=-1)
        neg_sim = -tf.keras.losses.cosine_similarity(anchor, negative, axis=-1)

        return K.maximum(neg_sim - pos_sim + margin, 0)

    return f