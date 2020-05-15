from tensorflow.keras import backend as K
import tensorflow as tf


def eucl_dist_pos(_, y_pred):
    vector_size = y_pred.shape[1] // 3
    anchor = y_pred[:, :vector_size]
    positive = y_pred[:, vector_size:2*vector_size]
    pos_dist = K.mean(K.square(anchor - positive), axis=1)
    return pos_dist

def eucl_dist_neg(_, y_pred):
    vector_size = y_pred.shape[1] // 3
    anchor = y_pred[:, :vector_size]
    negative = y_pred[:, 2*vector_size:]
    neg_dist = K.mean(K.square(anchor - negative), axis=1)
    return neg_dist

def cos_sim_pos(_, y_pred):
    vector_size = y_pred.shape[-1] // 3
    anchor_vec = y_pred[:, :vector_size]
    positive_vec = y_pred[:, vector_size:2*vector_size]
    d = tf.keras.losses.cosine_similarity(anchor_vec, positive_vec)
    return d

def cos_sim_neg(_, y_pred):
    vector_size = y_pred.shape[-1] // 3
    anchor_vec = y_pred[:, :vector_size]
    negative_vec = y_pred[:, 2*vector_size:]
    d = tf.keras.losses.cosine_similarity(anchor_vec, negative_vec)
    return d