import tensorflow as tf
import transformers
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input, GlobalAveragePooling1D, Concatenate, Dense
from tensorflow.keras.models import Model
from AES.layers.topk_self_attention import SingleSelfAttention


# https://colab.research.google.com/drive/1VgOTzr_VZNHkXh2z9IiTAcEgg5qr19y0#scrollTo=-JwxoYiyajM8 #
def triplet_loss_eucl(y_true, y_pred, margin=1):
    vector_size = y_pred.shape[-1] // 3
    anchor = y_pred[:, :vector_size]
    positive = y_pred[:, vector_size:2*vector_size]
    negative = y_pred[:, 2*vector_size:]
    pos_dist = K.mean(K.square(anchor - positive), axis=1)
    neg_dist = K.mean(K.square(anchor - negative), axis=1)
    return K.maximum(pos_dist - neg_dist + margin, 0.0)
    return loss

def eucl_dist_pos(y_true, y_pred):
    vector_size = y_pred.shape[1] // 3
    anchor = y_pred[:, :vector_size]
    positive = y_pred[:, vector_size:2*vector_size]
    pos_dist = K.mean(K.square(anchor - positive), axis=1)
    return pos_dist

def eucl_dist_neg(y_true, y_pred):
    vector_size = y_pred.shape[1] // 3
    anchor = y_pred[:, :vector_size]
    negative = y_pred[:, 2*vector_size:]
    neg_dist = K.mean(K.square(anchor - negative), axis=1)
    return neg_dist

def triplet_loss(y_true, y_pred, margin=1):
    VECTOR_SIZE = y_pred.shape[-1] // 3
    anchor_vec = y_pred[:, :VECTOR_SIZE]
    positive_vec = y_pred[:, VECTOR_SIZE:2*VECTOR_SIZE]
    negative_vec = y_pred[:, 2*VECTOR_SIZE:]
    d1 = tf.keras.losses.cosine_similarity(anchor_vec, positive_vec, axis=-1)
    d2 = tf.keras.losses.cosine_similarity(anchor_vec, negative_vec, axis=-1)
    return K.clip(d2 - d1 + margin, 0, None)

def cos_sim_pos(y_true, y_pred):
    VECTOR_SIZE = y_pred.shape[-1] // 3
    anchor_vec = y_pred[:, :VECTOR_SIZE]
    positive_vec = y_pred[:, VECTOR_SIZE:2*VECTOR_SIZE]
    d1 = tf.keras.losses.cosine_similarity(anchor_vec, positive_vec)
    return d1

def cos_sim_neg(y_true, y_pred):
    VECTOR_SIZE = y_pred.shape[-1] // 3
    anchor_vec = y_pred[:, :VECTOR_SIZE]
    negative_vec = y_pred[:, 2*VECTOR_SIZE:]
    d2 = tf.keras.losses.cosine_similarity(anchor_vec, negative_vec)
    return d2

n_max_sents = 23


xd = Input(shape=(None, 32))
xp = Input(shape=(None, 32))
xn = Input(shape=(None, 32))

fc = Dense(32, activation="linear")

hd = fc(xd)
hp = fc(xp)
hn = fc(xn)
#https://stackoverflow.com/questions/38548871/tensorflow-how-to-propagate-gradient-through-tf-gather #
sel_indices, candidate = SingleSelfAttention(0, None, n_max_sents)(hd)

hc = GlobalAveragePooling1D()(candidate)
hp = GlobalAveragePooling1D()(hp)
hn = GlobalAveragePooling1D()(hn)

o = Concatenate(axis=-1)([hc, hp, hn])

model = Model(inputs=[xd, xp, xn], outputs=o)
model_test = Model(inputs=[xd, xp, xn], outputs = [sel_indices, candidate])
model.compile(loss=triplet_loss_eucl, optimizer="adam", metrics=[eucl_dist_neg, eucl_dist_pos])





print(model.summary())
Xd = np.hstack((np.zeros((50000, 10, 32))+30,
                np.zeros((50000, 5, 32)),
                np.zeros((50000, 8, 32))+50))

Xp = np.hstack((np.zeros((50000, 10, 32))+90,
                np.zeros((50000, 5, 32)),
                np.zeros((50000, 8, 32))+15))

Xn = np.hstack((np.zeros((50000, 10, 32))+68,
                np.zeros((50000, 5, 32))+36,
                np.zeros((50000, 8, 32))+38))


pred = model_test.predict([Xd[:2], Xp[:2], Xn[:2]])
sel_ind = pred[0][0]
cand = pred[1][0]
print("BEFORE")
print(sel_ind)
print(cand)

model.fit([Xd, Xp, Xn], y=np.empty((50000,)), batch_size=32, epochs=1)


pred = model_test.predict([Xd[:2], Xp[:2], Xn[:2]])
sel_ind = pred[0][0]
cand = pred[1][0]
print("AFTER")
print(sel_ind)
print(cand)