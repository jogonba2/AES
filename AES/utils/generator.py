from AES.utils import utils as ut


class TrainGenerator:

    def __init__(self, dataset_file, tokenizer,
                 max_len_sent_doc, max_sents_doc,
                 max_len_sent_summ, max_sents_summ,
                 sent_split="<s>"):

        self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.max_len_sent_doc = max_len_sent_doc
        self.max_sents_doc = max_sents_doc
        self.max_len_sent_summ = max_len_sent_summ
        self.max_sents_summ = max_sents_summ
        self.sent_split = sent_split
        self.max_len_seq_doc = (self.max_len_sent_doc * self.max_sents_doc) + (self.max_sents_doc * 2)
        self.max_len_seq_summ = (self.max_len_sent_summ * self.max_sents_summ) + (self.max_sents_summ * 2)


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

                yield ({"doc_token_ids": doc_token_ids,
                        "doc_positions": doc_positions,
                        "doc_segments": doc_segments,
                        "doc_masks": doc_masks,
                        "summ_token_ids": summ_token_ids,
                        "summ_positions": summ_positions,
                        "summ_segments": summ_segments,
                        "summ_masks": summ_masks},
                        {"matching": 0, "selecting": 0})
