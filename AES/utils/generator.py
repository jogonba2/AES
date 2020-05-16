from AES.utils import utils as ut
from AES.utils.triplet_batching import TripletBatching
import numpy as np


# https://omoindrot.github.io/triplet-loss
class TrainGenerator:

    def __init__(self, dataset_file, batch_size, tokenizer,
                 max_len_sent_doc, max_sents_doc,
                 max_len_sent_summ, max_sents_summ,
                 random_sent_ordering=False, sent_split="<s>"):

        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len_sent_doc = max_len_sent_doc
        self.max_sents_doc = max_sents_doc
        self.max_len_sent_summ = max_len_sent_summ
        self.max_sents_summ = max_sents_summ
        self.random_sent_ordering = random_sent_ordering
        self.sent_split = sent_split
        self.max_len_seq_doc = (self.max_len_sent_doc * self.max_sents_doc) + (self.max_sents_doc * 2)
        self.max_len_seq_summ = (self.max_len_sent_summ * self.max_sents_summ) + (self.max_sents_summ * 2)
        self.triplet_batcher = TripletBatching(self.batch_size, self.max_len_seq_doc,
                                               self.max_len_seq_summ, self.random_sent_ordering)

    def generator(self):
        while True:
            fr = open(self.dataset_file, "r", encoding="utf8")
            fr.readline() # Skip header

            for line in fr.readlines():
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

                (summ_token_ids, summ_positions,
                 summ_segments, summ_masks) = ut.prepare_inputs(summ_sents,
                                                                self.tokenizer,
                                                                self.max_len_sent_summ,
                                                                self.max_sents_summ)


                batch = self.triplet_batcher.update(doc_token_ids, doc_positions,
                                                    doc_segments, doc_masks,
                                                    summ_token_ids, summ_positions,
                                                    summ_segments, summ_masks)


                if batch is not None:
                    yield (batch, np.empty(self.batch_size))