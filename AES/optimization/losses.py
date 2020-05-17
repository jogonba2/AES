from tensorflow.keras import backend as K
import tensorflow as tf


# Triplet loss among document, summary and candidate #
def match_loss_cos(margin=1):
    def f(_, y_pred):
        vector_size = y_pred.shape[-1] // 3
        doc_repr = y_pred[:, :vector_size]
        summ_repr = y_pred[:, vector_size:2*vector_size]
        cand_repr = y_pred[:, 2*vector_size:]
        sim_doc_cand = tf.keras.losses.cosine_similarity(doc_repr, cand_repr, axis=-1)
        sim_doc_summ = tf.keras.losses.cosine_similarity(doc_repr, summ_repr, axis=-1)
        return K.maximum(sim_doc_cand - sim_doc_summ + margin, 0.0)
    return f

# Cosine distance between candidate and summary #
def select_loss_cos(_, y_pred):
    vector_size = y_pred.shape[-1] // 2
    summ_repr = y_pred[:, :vector_size]
    cand_repr = y_pred[:, vector_size:]
    sim_summ_cand = tf.keras.losses.cosine_similarity(summ_repr, cand_repr, axis=-1)
    return 1. - sim_summ_cand