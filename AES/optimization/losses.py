from tensorflow.keras import backend as K
import tensorflow as tf


def triplet_loss_eucl(margin=1):
    def f(_, y_pred):
        vector_size = y_pred.shape[-1] // 3
        anchor = y_pred[:, :vector_size]
        positive = y_pred[:, vector_size:2*vector_size]
        negative = y_pred[:, 2*vector_size:]
        pos_dist = K.mean(K.square(anchor - positive), axis=1)
        neg_dist = K.mean(K.square(anchor - negative), axis=1)
        return K.maximum(pos_dist - neg_dist + margin, 0.0)
        return loss
    return f

def triplet_loss_cos(margin=1):
    def f(_, y_pred, ):
        vector_size = y_pred.shape[-1] // 3
        anchor_vec = y_pred[:, :vector_size]
        positive_vec = y_pred[:, vector_size:2*vector_size]
        negative_vec = y_pred[:, 2*vector_size:]
        pos_sim = tf.keras.losses.cosine_similarity(anchor_vec, positive_vec, axis=-1)
        neg_sim = tf.keras.losses.cosine_similarity(anchor_vec, negative_vec, axis=-1)
        return K.maximum(neg_sim - pos_sim + margin, 0.0)
    return f


