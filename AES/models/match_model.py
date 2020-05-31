import AES.optimization.losses as losses_module
import AES.optimization.metrics as metrics_module
from AES.optimization.lr_annealing import Noam
import tensorflow as tf
import transformers
import numpy as np


class MatchModel():

    def __init__(self, model_name, shortcut_weights,
                 max_len_sent_doc, max_sents_doc,
                 max_len_sent_summ, max_sents_summ,
                 noam_annealing=False, noam_params=None,
                 learning_rate=1e-3,
                 loss_1="cosine_triplet_loss",
                 loss_2="cosine_ranking_loss",
                 margin_1=1.,
                 margin_2=1.,
                 metrics={"outputs": ["cosine_similarity_pos_pair",
                                      "cosine_similarity_neg_pair"]},
                 avg_att_layers="all", train=True):

        self.model_name = model_name
        self.shortcut_weights = shortcut_weights
        self.max_len_sent_doc = max_len_sent_doc
        self.max_sents_doc = max_sents_doc
        self.max_len_sent_summ = max_len_sent_summ
        self.max_sents_summ = max_sents_summ
        self.max_len_seq_doc = (self.max_len_sent_doc * self.max_sents_doc) + (self.max_sents_doc * 2)
        self.max_len_seq_summ = (self.max_len_sent_summ * self.max_sents_summ) + (self.max_sents_summ * 2)
        self.learning_rate = learning_rate
        self.losses = {"outputs": getattr(losses_module, loss_1)(margin_1),
                       "outputs2": getattr(losses_module, loss_2)(margin_2)}
        self.metrics = {"outputs": [getattr(metrics_module, metric) for metric in metrics["outputs"]],
                        "outputs2": getattr(metrics_module, metrics["outputs"][1])}
        self.avg_att_layers = avg_att_layers
        self.noam_annealing = noam_annealing
        self.noam_params = noam_params
        self.optimizer = self.build_optimizer()
        self.train = train

    def _get_sentence_scores(self, doc_attentions):
        if self.avg_att_layers == "all":
            layers = [layer for layer in doc_attentions]
        else:
            layers = [doc_attentions[l] for l in self.avg_att_layers]

        # Average encoder attentions #
        if len(layers) == 1:
            attentions_doc = layers[0]
        else:
            attentions_doc = tf.keras.layers.Add()(layers) / len(layers)

        # Get CLSs #
        attentions_doc = attentions_doc[:, :,
                                        0:self.max_len_seq_doc:self.max_len_sent_doc+2,
                                        0:self.max_len_seq_doc:self.max_len_sent_doc+2]

        # Average attention heads #
        avg_att_heads_doc = tf.keras.backend.mean(attentions_doc, axis=1)

        # Average CLSs attentions #
        att_sent_scores = tf.keras.backend.mean(avg_att_heads_doc, axis=1)

        # Normalize attentions #
        att_sent_scores = tf.keras.layers.Activation("softmax")(att_sent_scores)

        return att_sent_scores

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

        with tf.device("/GPU:1"):
            input_ids = {"document": tf.keras.layers.Input(shape=(self.max_len_seq_doc,),
                                                           dtype=tf.int32, name="doc_token_ids"),
                         "positive": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                           dtype=tf.int32, name="pos_token_ids"),
                         "candidate_i": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                           dtype=tf.int32, name="candi_token_ids"),
                         "candidate_j": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                              dtype=tf.int32, name="candj_token_ids")
                         }

            input_masks = {"document": tf.keras.layers.Input(shape=(self.max_len_seq_doc,),
                                                             dtype=tf.int32, name="doc_masks"),
                           "positive": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                             dtype=tf.int32, name="pos_masks"),
                           "candidate_i": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                             dtype=tf.int32, name="candi_masks"),
                           "candidate_j": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                                dtype=tf.int32, name="candj_masks")
                           }

            input_segments = {"document": tf.keras.layers.Input(shape=(self.max_len_seq_doc,),
                                                                dtype=tf.int32, name="doc_segments"),
                              "positive": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                                dtype=tf.int32, name="pos_segments"),
                              "candidate_i": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                                dtype=tf.int32, name="candi_segments"),
                              "candidate_j": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                                   dtype=tf.int32, name="candj_segments")
                              }

            input_positions = {"document": tf.keras.layers.Input(shape=(self.max_len_seq_doc,),
                                                                 dtype=tf.int32, name="doc_positions"),
                               "positive": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                                 dtype=tf.int32, name="pos_positions"),
                               "candidate_i": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                                 dtype=tf.int32, name="candi_positions"),
                               "candidate_j": tf.keras.layers.Input(shape=(self.max_len_seq_summ,),
                                                                    dtype=tf.int32, name="candj_positions")
                               }


            # Select HuggingFace model
            sent_encoder = getattr(transformers,
                                   self.model_name).from_pretrained(self.shortcut_weights,
                                                                    output_attentions=True).layers[0]

            # Add position embeddings until cover the max_seq_len #
            self._add_position_embeddings(sent_encoder.embeddings.position_embeddings)

            # Document/Candidate Branch #
            doc_outs = sent_encoder([input_ids["document"],
                                     input_masks["document"],
                                     input_segments["document"],
                                     input_positions["document"]])

            att_sent_scores = self._get_sentence_scores(doc_outs[2])

            encoded_doc = doc_outs[0]


            encoded_doc_sents = tf.gather(encoded_doc,
                                          [_ for _ in range(0, self.max_len_seq_doc,
                                                            self.max_len_sent_doc+2)],
                                          axis=1)

            document_repr = tf.keras.layers.GlobalAveragePooling1D()(encoded_doc_sents)


            # Positive Branch #
            pos_outs = sent_encoder([input_ids["positive"],
                                     input_masks["positive"],
                                     input_segments["positive"],
                                     input_positions["positive"]])

            encoded_positive = pos_outs[0]



            encoded_positive_sents = tf.gather(encoded_positive,
                                          [_ for _ in range(0, self.max_len_seq_summ,
                                                            self.max_len_sent_summ+2)],
                                          axis=1)

            positive_repr = tf.keras.layers.GlobalAveragePooling1D()(encoded_positive_sents)

            # Candidate i Branch #
            candi_outs = sent_encoder([input_ids["candidate_i"],
                                     input_masks["candidate_i"],
                                     input_segments["candidate_i"],
                                     input_positions["candidate_i"]])

            encoded_candi = candi_outs[0]

            encoded_candi_sents = tf.gather(encoded_candi,
                                          [_ for _ in range(0, self.max_len_seq_summ,
                                                            self.max_len_sent_summ+2)],
                                          axis=1)

            candi_repr = tf.keras.layers.GlobalAveragePooling1D()(encoded_candi_sents)

            # Candidate j Branch #
            candj_outs = sent_encoder([input_ids["candidate_j"],
                                     input_masks["candidate_j"],
                                     input_segments["candidate_j"],
                                     input_positions["candidate_j"]])

            encoded_candj = candj_outs[0]


            encoded_candj_sents = tf.gather(encoded_candj,
                                          [_ for _ in range(0, self.max_len_seq_summ,
                                                            self.max_len_sent_summ+2)],
                                          axis=1)

            candj_repr = tf.keras.layers.GlobalAveragePooling1D()(encoded_candj_sents)


            # Concatenate (document, reference and candidate_i) for computing the loss_1 #
            outputs = tf.keras.layers.Concatenate(axis=-1, name="outputs")([document_repr,
                                                                            positive_repr,
                                                                            candi_repr])

            # Concatenate (document, candidate_i and candidate_j) for computing the loss_2 #
            outputs2 = tf.keras.layers.Concatenate(axis=-1, name="outputs2")([document_repr,
                                                                              candi_repr,
                                                                              candj_repr])

            # Define models #
            self.model = tf.keras.Model(inputs=[input_ids["document"],
                                                       input_masks["document"],
                                                        input_segments["document"],
                                                        input_positions["document"],
                                                        input_ids["positive"],
                                                        input_masks["positive"],
                                                        input_segments["positive"],
                                                        input_positions["positive"],
                                                        input_ids["candidate_i"],
                                                        input_masks["candidate_i"],
                                                        input_segments["candidate_i"],
                                                        input_positions["candidate_i"],
                                                        input_ids["candidate_j"],
                                                        input_masks["candidate_j"],
                                                        input_segments["candidate_j"],
                                                        input_positions["candidate_j"]],
                                                outputs=[outputs, outputs2])

            self.pos_repr_model = tf.keras.Model(inputs=[input_ids["positive"],
                                                 input_masks["positive"],
                                                 input_segments["positive"],
                                                 input_positions["positive"]],
                                                 outputs=positive_repr)

            self.candi_repr_model = tf.keras.Model(inputs=[input_ids["candidate_i"],
                                                 input_masks["candidate_i"],
                                                 input_segments["candidate_i"],
                                                 input_positions["candidate_i"]],
                                                 outputs=candi_repr)

            self.candj_repr_model = tf.keras.Model(inputs=[input_ids["candidate_j"],
                                                 input_masks["candidate_j"],
                                                 input_segments["candidate_j"],
                                                 input_positions["candidate_j"]],
                                                 outputs=candj_repr)

            self.selector_model = tf.keras.Model(inputs=[input_ids["document"],
                                                 input_masks["document"],
                                                 input_segments["document"],
                                                 input_positions["document"]],
                                                 outputs=att_sent_scores)

    def compile(self):
        assert self.model
        self.model.compile(optimizer=self.optimizer,
                           loss=self.losses,
                           metrics=self.metrics)

    def build_optimizer(self):
        if self.noam_annealing:
            lr_noam = Noam(self.noam_params["warmup_steps"],
                           self.noam_params["hidden_dims"])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_noam)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        return optimizer

    def get_optimizer(self):
        return self.optimizer

    def save_model(self, model, path_save_weights):
        model.save_weights(path_save_weights)

