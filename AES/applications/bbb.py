import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# script parameters
BATCH_SIZE = 8
EVAL_BATCH_SIZE = BATCH_SIZE
USE_XLA = False
USE_AMP = False

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

# Load model from pretrained model/vocabulary
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=1000)


def gen():
    for x1, x2, x3, y in zip(np.zeros((80, 512), dtype=np.int32),
                          np.zeros((80, 512), dtype=np.int32),
                          np.zeros((80, 512), dtype=np.int32),
                          np.zeros((80, 1000), dtype=np.int32)):
        yield ({'input_ids': x1,
                'attention_mask': x2,
                'token_type_ids': x3}, y)


# Prepare dataset as tf.Dataset from generator
dataset = tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int32),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([None])))