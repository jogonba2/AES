import tensorflow as tf
from tensorflow.keras import backend as K


def cosine_distance(x, y):
    return K.batch_dot(tf.nn.l2_normalize(x, 1),
                       tf.nn.l2_normalize(y, 1))