from AES.layers import self_attention as self_attention
import AES.optimization.losses as losses_module
import AES.optimization.metrics as metrics_module
import tensorflow as tf
import transformers
import numpy as np


# https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer #
# https://huggingface.co/transformers/model_doc/bert.html #
class AESModel():

    def __init__(self, model_name, shortcut_weights,
                 max_len_sent_doc, max_sents_doc,
                 max_len_sent_summ, max_sents_summ, att_layer,
                 att_params, optimizer, loss="triplet_loss_cos",
                 margin=1., metrics=["cos_sim_pos", "cos_sim_neg"]):

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
        self.optimizer = optimizer
        self.loss = getattr(losses_module, loss)
        self.margin = margin
        self.metrics = [getattr(metrics_module, metric) for metric in metrics]

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
                                                          dtype=tf.int32),
                     "pos_summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                          dtype=tf.int32),
                     "neg_summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                          dtype=tf.int32)}

        input_masks = {"document": tf.keras.layers.Input((self.max_len_seq_doc,),
                                                            dtype=tf.int32),
                       "pos_summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                            dtype=tf.int32),
                       "neg_summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                            dtype=tf.int32)}

        input_segments = {"document": tf.keras.layers.Input((self.max_len_seq_doc,),
                                                            dtype=tf.int32),
                          "pos_summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                               dtype=tf.int32),
                          "neg_summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                               dtype=tf.int32)}

        input_positions = {"document": tf.keras.layers.Input((self.max_len_seq_doc,),
                                                             dtype=tf.int32),
                          "pos_summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                               dtype=tf.int32),
                          "neg_summary": tf.keras.layers.Input((self.max_len_seq_summ,),
                                                               dtype=tf.int32)}

        # Select HuggingFace model #
        sent_encoder = getattr(transformers,
                               self.model_name).from_pretrained(self.shortcut_weights)

        # Add position embeddings until cover the max_seq_len #
        self._add_position_embeddings(sent_encoder.bert.embeddings.position_embeddings)

        # Encode sentences of document, pos summary and neg summary#
        encoded_doc_sents = sent_encoder(input_ids["document"],
                                         token_type_ids=input_segments["document"],
                                         position_ids=input_positions["document"],
                                         attention_mask = input_masks["document"])[0]

        encoded_pos_sents = sent_encoder(input_ids["pos_summary"],
                                         token_type_ids=input_segments["pos_summary"],
                                         position_ids=input_positions["pos_summary"],
                                         attention_mask = input_masks["pos_summary"])[0]

        encoded_neg_sents = sent_encoder(input_ids["neg_summary"],
                                         token_type_ids=input_segments["neg_summary"],
                                         position_ids=input_positions["neg_summary"],
                                         attention_mask = input_masks["neg_summary"])[0]

        encoded_cls = tf.gather(encoded_doc_sents,
                                [_ for _ in range(0, self.max_len_seq_doc,
                                                  self.max_len_sent_doc+2)],
                                axis=1)

        sel_indices, candidate = self.att_layer(self.att_params)(encoded_cls)

        candidate_repr = tf.keras.layers.GlobalAveragePooling1D()(candidate)
        pos_repr = tf.keras.layers.GlobalAveragePooling1D()(encoded_pos_sents)
        neg_repr = tf.keras.layers.GlobalAveragePooling1D()(encoded_neg_sents)

        output = tf.keras.layers.Concatenate(axis=-1)([candidate_repr,
                                                       pos_repr,
                                                       neg_repr])

        # Define model #
        self.tr_model = tf.keras.Model(inputs=[input_ids["document"],
                                               input_masks["document"],
                                               input_segments["document"],
                                               input_positions["document"],
                                               input_ids["pos_summary"],
                                               input_masks["pos_summary"],
                                               input_segments["pos_summary"],
                                               input_positions["pos_summary"],
                                               input_ids["neg_summary"],
                                               input_masks["neg_summary"],
                                               input_segments["neg_summary"],
                                               input_positions["neg_summary"]],
                                       outputs=[output])

        self.summ_model = tf.keras.Model(inputs=[input_ids["document"],
                                                 input_masks["document"],
                                                 input_segments["document"],
                                                 input_positions["document"],
                                                 input_ids["pos_summary"],
                                                 input_masks["pos_summary"],
                                                 input_segments["pos_summary"],
                                                 input_positions["pos_summary"],
                                                 input_ids["neg_summary"],
                                                 input_masks["neg_summary"],
                                                 input_segments["neg_summary"],
                                                 input_positions["neg_summary"]],
                                         outputs=[sel_indices])


    def compile(self):
        assert self.tr_model
        self.tr_model.compile(optimizer=self.optimizer,
                              loss=self.loss(self.margin),
                              metrics=self.metrics)

    def save_model(self):
        pass

    def load_model(self):
        pass