from AES.layers import self_attention as self_attention
import AES.optimization.losses as losses_module
import AES.optimization.metrics as metrics_module
from tensorflow.keras import backend as K
import tensorflow as tf
import transformers
import numpy as np


# https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer #
# https://huggingface.co/transformers/model_doc/bert.html #
class AESModel():

    def __init__(self, model_name, shortcut_weights,
                 max_len_sent_doc, max_sents_doc,
                 max_len_sent_summ, max_sents_summ, att_layer,
                 att_params, match_loss="match_loss_cos",
                 margin=1., select_loss="select_loss_cos",
                 metrics={"matching": ["cos_sim_doc_summ",
                                       "cos_sim_doc_cand"],
                          "selecting": "cos_sim_summ_cand"}):

        self.model_name = model_name
        self.shortcut_weights = shortcut_weights
        self.max_len_sent_doc = max_len_sent_doc
        self.max_sents_doc = max_sents_doc
        self.max_len_sent_summ = max_len_sent_summ
        self.max_sents_summ = max_sents_summ
        self.max_len_seq_doc = (self.max_len_sent_doc * self.max_sents_doc) + (self.max_sents_doc * 2)
        self.max_len_seq_summ = (self.max_len_sent_summ * self.max_sents_summ) + (self.max_sents_summ * 2)
        self.att_layer = getattr(self_attention, att_layer)
        self.att_params = att_params
        self.losses = {"matching": getattr(losses_module, match_loss)(margin),
                       "selecting": getattr(losses_module, select_loss)}
        self.metrics = {"matching": [getattr(metrics_module, metric) for metric in metrics["matching"]],
                        "selecting": getattr(metrics_module, metrics["selecting"])}

    # Add more position embeddings for sequences larger than 512 tokens #
    def _add_position_embeddings(self, position_embeddings):
        current_weights = position_embeddings.get_weights()[0]
        current_positions = len(current_weights)
        dim_emb = current_weights[0].shape[-1]
        # We avoided max(max_len_seq_doc, max_len_seq_summ) -> len_seq_doc always >>> len_seq_summ #
        if self.max_len_seq_doc > current_positions:
            rows_to_add = self.max_len_seq_doc - current_positions
            new_rows = np.random.random((rows_to_add, dim_emb))
            updated_weights = np.vstack((current_weights, new_rows))
            position_embeddings.input_dim = self.max_len_seq_doc
            position_embeddings.embeddings = tf.keras.backend.variable(updated_weights)
            position_embeddings.set_weights([updated_weights])

    def build(self):
        input_ids = {"document": tf.keras.layers.Input((self.max_len_seq_doc,),
                                                        dtype=tf.int32, name="doc_token_ids"),
                     "summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                       dtype=tf.int32, name="summ_token_ids")}

        input_masks = {"document": tf.keras.layers.Input((self.max_len_seq_doc,),
                                                          dtype=tf.int32, name="doc_masks"),
                       "summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                         dtype=tf.int32, name="summ_masks")}

        input_segments = {"document": tf.keras.layers.Input((self.max_len_seq_doc,),
                                                             dtype=tf.int32, name="doc_segments"),
                          "summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                            dtype=tf.int32, name="summ_segments")}

        input_positions = {"document": tf.keras.layers.Input((self.max_len_seq_doc,),
                                                              dtype=tf.int32, name="doc_positions"),
                           "summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                             dtype=tf.int32, name="summ_positions")}

        # Select HuggingFace model #
        sent_encoder = getattr(transformers, self.model_name).from_pretrained(self.shortcut_weights)

        # Add position embeddings until cover the max_seq_len #
        self._add_position_embeddings(sent_encoder.bert.embeddings.position_embeddings)

        # Encode document and reference summary #
        encoded_doc = sent_encoder(input_ids["document"],
                                   token_type_ids=input_segments["document"],
                                   position_ids=input_positions["document"],
                                   attention_mask = input_masks["document"])[0]

        encoded_summ = sent_encoder(input_ids["summary"],
                                    token_type_ids=input_segments["summary"],
                                    position_ids=input_positions["summary"],
                                    attention_mask = input_masks["summary"])[0]

        # Gather the CLS tokens for representing the sentences of document and reference summary #
        encoded_doc_sents = tf.gather(encoded_doc,
                                      [_ for _ in range(0, self.max_len_seq_doc,
                                                        self.max_len_sent_doc+2)],
                                      axis=1)

        encoded_summ_sents = tf.gather(encoded_summ,
                                      [_ for _ in range(0, self.max_len_seq_summ,
                                                        self.max_len_sent_summ+2)],
                                      axis=1)

        # From the CLS tokens of the document, build the candidate summary #
        alpha, candidate = self.att_layer(self.att_params)(encoded_doc_sents)

        # Averaging sentence representations of document (unweighted), reference summary (unweighted)
        # and candidate (weighted by alpha)
        doc_repr = tf.keras.layers.GlobalAveragePooling1D()(encoded_doc_sents)
        cand_repr = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1))(candidate)
        summ_repr = tf.keras.layers.GlobalAveragePooling1D()(encoded_summ_sents)


        # Concatenate all the outputs for computing the matching loss #
        matching_outputs = tf.keras.layers.Concatenate(axis=-1, name="matching")([doc_repr,
                                                                                 summ_repr,
                                                                                 cand_repr])

        # Concatenate summ_repr and cand_repr for computing the selection loss #
        selecting_outputs = tf.keras.layers.Concatenate(axis=-1, name="selecting")([summ_repr,
                                                                                    cand_repr])

        # Define model #
        self.model = tf.keras.Model(inputs=[input_ids["document"],
                                            input_masks["document"],
                                            input_segments["document"],
                                            input_positions["document"],
                                            input_ids["summary"],
                                            input_masks["summary"],
                                            input_segments["summary"],
                                            input_positions["summary"]],
                                       outputs=[matching_outputs,
                                                selecting_outputs])


        self.summ_model = tf.keras.Model(inputs=[input_ids["document"],
                                                 input_masks["document"],
                                                 input_segments["document"],
                                                 input_positions["document"],
                                                 input_ids["summary"],
                                                 input_masks["summary"],
                                                 input_segments["summary"],
                                                 input_positions["summary"]],
                                         outputs=[alpha])
        

    def compile(self):
        assert self.model
        self.model.compile(optimizer="adam",
                           loss=self.losses,
                           metrics=self.metrics)

    def save_model(self, path_weights):
        assert self.model
        self.model.save_weights(path_weights)

    def load_model(self):
        pass