import tensorflow as tf


def cos_sim_doc_summ(_, y_pred):
    vector_size = y_pred.shape[-1] // 3
    doc_repr = y_pred[:, :vector_size]
    summ_repr = y_pred[:, vector_size:2 * vector_size]
    sim_doc_summ = tf.keras.losses.cosine_similarity(doc_repr, summ_repr, axis=-1)
    return sim_doc_summ

def cos_sim_doc_cand(_, y_pred):
    vector_size = y_pred.shape[-1] // 3
    doc_repr = y_pred[:, :vector_size]
    cand_repr = y_pred[:, 2 * vector_size:]
    sim_doc_cand = tf.keras.losses.cosine_similarity(doc_repr, cand_repr, axis=-1)
    return sim_doc_cand

def cos_sim_summ_cand(_, y_pred):
    vector_size = y_pred.shape[-1] // 2
    summ_repr = y_pred[:, :vector_size]
    cand_repr = y_pred[:, vector_size:]
    sim_summ_cand = tf.keras.losses.cosine_similarity(summ_repr, cand_repr, axis=-1)
    return sim_summ_cand