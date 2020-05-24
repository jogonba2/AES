from AES.utils import utils as ut
from AES.optimization import metrics
from random import shuffle
import numpy as np


class SelectGenerator:

    def __init__(self, dataset_file, tokenizer,
                 triplet_mining_model, batch_size,
                 max_len_sent_doc, max_sents_doc,
                 max_len_sent_summ, max_sents_summ,
                 sent_split="<s>", train=True,
                 ranking_triplet_func="pairwise_cosine_similarity_rank"):

        self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.triplet_mining_model = triplet_mining_model
        self.batch_size = batch_size
        self.max_len_sent_doc = max_len_sent_doc
        self.max_sents_doc = max_sents_doc
        self.max_len_sent_summ = max_len_sent_summ
        self.max_sents_summ = max_sents_summ
        self.sent_split = sent_split
        self.train = train
        self.max_len_seq_doc = (self.max_len_sent_doc * self.max_sents_doc) + (self.max_sents_doc * 2)
        self.max_len_seq_summ = (self.max_len_sent_summ * self.max_sents_summ) + (self.max_sents_summ * 2)
        self.ranking_triplet_func = getattr(metrics, ranking_triplet_func)

        self.batch_doc_token_ids = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_positive_token_ids = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_negative_token_ids = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")

        self.batch_doc_positions = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_positive_positions = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_negative_positions = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")

        self.batch_doc_segments = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_positive_segments = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_negative_segments = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")

        self.batch_doc_masks = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_positive_masks = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_negative_masks = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")

        self.batch_samples = 0


    def _build_neg_triplets(self, embeddings):
        candidates = embeddings[:, :embeddings.shape[-1]//2]
        positives = embeddings[:, embeddings.shape[-1]//2:]
        ranking = self.ranking_triplet_func(candidates, positives)
        # print(ranking)
        for i in range(embeddings.shape[0]):
            best_fitted = ranking[i][0]
            self.batch_negative_token_ids[i] = self.batch_positive_token_ids[best_fitted]
            self.batch_negative_positions[i] = self.batch_positive_positions[best_fitted]
            self.batch_negative_segments[i] = self.batch_positive_segments[best_fitted]
            self.batch_negative_masks[i] = self.batch_positive_masks[best_fitted]

    def generator(self):
        while True:
            fr = open(self.dataset_file, "r", encoding="utf8")
            fr.readline() # Skip header
            lines = fr.readlines()
            shuffle(lines)

            for line in lines:

                if self.batch_samples % self.batch_size == 0 and self.batch_samples > 0:
                    self.batch_samples = 0

                    # Get embeddings for triplet generator #
                    embeddings = self.triplet_mining_model.predict({"doc_token_ids": self.batch_doc_token_ids,
                                                                    "doc_positions": self.batch_doc_positions,
                                                                    "doc_segments": self.batch_doc_segments,
                                                                    "doc_masks": self.batch_doc_masks,
                                                                    "pos_token_ids": self.batch_positive_token_ids,
                                                                    "pos_positions": self.batch_positive_positions,
                                                                    "pos_segments": self.batch_positive_segments,
                                                                    "pos_masks": self.batch_positive_masks})

                    # Triplet building #
                    self._build_neg_triplets(embeddings)
                    if self.train:
                        yield ({"doc_token_ids": self.batch_doc_token_ids,
                                "doc_positions": self.batch_doc_positions,
                                "doc_segments": self.batch_doc_segments,
                                "doc_masks": self.batch_doc_masks,
                                "pos_token_ids": self.batch_positive_token_ids,
                                "pos_positions": self.batch_positive_positions,
                                "pos_segments": self.batch_positive_segments,
                                "pos_masks": self.batch_positive_masks,
                                "neg_token_ids": self.batch_negative_token_ids,
                                "neg_positions": self.batch_negative_positions,
                                "neg_segments": self.batch_negative_segments,
                                "neg_masks": self.batch_negative_masks},
                                {"outputs": np.zeros(self.batch_size, dtype="int32")})
                    else:
                        yield {"doc_token_ids": self.batch_doc_token_ids,
                                "doc_positions": self.batch_doc_positions,
                                "doc_segments": self.batch_doc_segments,
                                "doc_masks": self.batch_doc_masks,
                                "pos_token_ids": self.batch_positive_token_ids,
                                "pos_positions": self.batch_positive_positions,
                                "pos_segments": self.batch_positive_segments,
                                "pos_masks": self.batch_positive_masks,
                                "neg_token_ids": self.batch_negative_token_ids,
                                "neg_positions": self.batch_negative_positions,
                                "neg_segments": self.batch_negative_segments,
                                "neg_masks": self.batch_negative_masks}

                doc, summ = line.split("\t")
                doc_sents = ut.preprocess_text(doc, self.sent_split)
                summ_sents = ut.preprocess_text(summ, self.sent_split)

                if not doc or not summ:
                    continue

                (doc_token_ids, doc_positions,
                 doc_segments, doc_masks) = ut.prepare_inputs(doc_sents,
                                                              self.tokenizer,
                                                              self.max_len_sent_doc,
                                                              self.max_sents_doc)

                (positive_token_ids, positive_positions,
                 positive_segments, positive_masks) = ut.prepare_inputs(summ_sents,
                                                                self.tokenizer,
                                                                self.max_len_sent_summ,
                                                                self.max_sents_summ)

                self.batch_doc_token_ids[self.batch_samples] = doc_token_ids
                self.batch_positive_token_ids[self.batch_samples] = positive_token_ids
                self.batch_doc_positions[self.batch_samples] = doc_positions
                self.batch_positive_positions[self.batch_samples] = positive_positions
                self.batch_doc_segments[self.batch_samples] = doc_segments
                self.batch_positive_segments[self.batch_samples] = positive_segments
                self.batch_doc_masks[self.batch_samples] = doc_masks
                self.batch_positive_masks[self.batch_samples] = positive_masks

                self.batch_samples += 1


class SummarizeSelectGenerator:

    def __init__(self, dataset_file, tokenizer,
                 batch_size, max_len_sent_doc,
                 max_sents_doc, sent_split="<s>"):

        self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len_sent_doc = max_len_sent_doc
        self.max_sents_doc = max_sents_doc
        self.sent_split = sent_split
        self.max_len_seq_doc = (self.max_len_sent_doc * self.max_sents_doc) + (self.max_sents_doc * 2)

        self.batch_doc_token_ids = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_doc_positions = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_doc_segments = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_doc_masks = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")

        self.doc_sents = []
        self.summ_sents = []

        self.batch_samples = 0

    def generator(self):
        while True:
            fr = open(self.dataset_file, "r", encoding="utf8")
            fr.readline() # Skip header

            for line in fr.readlines():

                if self.batch_samples % self.batch_size == 0 and self.batch_samples > 0:
                    self.batch_samples = 0
                    yield ({"doc_token_ids": self.batch_doc_token_ids,
                            "doc_positions": self.batch_doc_positions,
                            "doc_segments": self.batch_doc_segments,
                            "doc_masks": self.batch_doc_masks},
                            [self.doc_sents, self.summ_sents])
                    self.doc_sents = []
                    self.summ_sents = []

                doc, summ = line.split("\t")

                doc_sents = ut.preprocess_text(doc, self.sent_split)
                summ_sents = ut.preprocess_text(summ, self.sent_split)

                self.doc_sents.append(doc_sents)
                self.summ_sents.append(summ_sents)

                if not doc or not summ:
                    continue

                (doc_token_ids, doc_positions,
                 doc_segments, doc_masks) = ut.prepare_inputs(doc_sents,
                                                              self.tokenizer,
                                                              self.max_len_sent_doc,
                                                              self.max_sents_doc)


                self.batch_doc_token_ids[self.batch_samples] = doc_token_ids
                self.batch_doc_positions[self.batch_samples] = doc_positions
                self.batch_doc_segments[self.batch_samples] = doc_segments
                self.batch_doc_masks[self.batch_samples] = doc_masks

                self.batch_samples += 1
