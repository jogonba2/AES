from AES.utils import utils as ut
from AES.optimization import metrics
from random import shuffle
from pythonrouge.pythonrouge import Pythonrouge
from rouge import Rouge
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

                if not doc_sents or not summ_sents:
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

            if not doc_sents or not summ_sents:
                continue

            self.doc_sents.append(doc_sents)
            self.summ_sents.append(summ_sents)

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




# GENERATORS FOR MATCH MODEL #

class MatchGenerator:

    def __init__(self, dataset_file, tokenizer,
                 aes, k_max, avg_att_layers,
                 ngram_blocking, batch_size, max_len_sent_doc,
                 max_sents_doc, max_len_sent_summ,
                 max_sents_summ, sent_split="<s>",
                 train=True):

        self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.k_max = k_max
        self.avg_att_layers = avg_att_layers
        self.ngram_blocking = ngram_blocking
        self.selector_model = aes.selector_model
        self.batch_size = batch_size
        self.max_len_sent_doc = max_len_sent_doc
        self.max_sents_doc = max_sents_doc
        self.max_len_sent_summ = max_len_sent_summ
        self.max_sents_summ = max_sents_summ
        self.sent_split = sent_split
        self.train = train
        self.max_len_seq_doc = (self.max_len_sent_doc * self.max_sents_doc) + (self.max_sents_doc * 2)
        self.max_len_seq_summ = (self.max_len_sent_summ * self.max_sents_summ) + (self.max_sents_summ * 2)

        self.batch_doc_token_ids = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_positive_token_ids = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_candi_token_ids = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_candj_token_ids = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")

        self.batch_doc_positions = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_positive_positions = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_candi_positions = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_candj_positions = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")

        self.batch_doc_segments = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_positive_segments = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_candi_segments = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_candj_segments = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")

        self.batch_doc_masks = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_positive_masks = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_candi_masks = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_candj_masks = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")

        # Se pasa como y_true a la loss
        self.batch_diffs_ranking = np.zeros((self.batch_size,), dtype="float32")

        self.batch_samples = 0
        self.bucket_max_size = 30000
        self._rouge = Rouge()


    def generator(self):

        # Empezar generador #
        while True:
            fr = open(self.dataset_file, "r", encoding="utf8")
            fr.readline()  # Skip header
            lines = fr.readlines()
            shuffle(lines)

            buckets = {}

            for doc_index, line in enumerate(lines):

                # Separar documento y referencia #
                doc, summ = line.split("\t")
                doc_sents = ut.preprocess_text(doc, self.sent_split)
                summ_sents = ut.preprocess_text(summ, self.sent_split)

                # Comprobar que documento y referencia son validos #
                if not doc_sents or not summ_sents:
                    continue

                n_doc_sents = len(doc_sents)
                if n_doc_sents < self.k_max:
                    continue

                # Preparo documento y referencia #
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

                # Añadir documento y referencia al bucket #
                buckets[doc_index] = {"document": {"token_ids": doc_token_ids,
                                                   "positions": doc_positions,
                                                   "segments": doc_segments,
                                                   "masks": doc_masks,
                                                   "sents": doc_sents},
                                       "reference": {"token_ids": positive_token_ids,
                                                    "positions": positive_positions,
                                                    "segments": positive_segments,
                                                    "masks": positive_masks,
                                                    "sents": np.array(summ_sents)},
                                      "candidates": {},
                                      "combination_indices": [],
                                      "ranking": []
                                      }

                # Si se ha llenado el bucket, ya tenemos todos los documentos y referencias #
                if len(buckets) == self.bucket_max_size:
                    # Batchify de los documentos del bucket #
                    documents_batch = ut.get_batch(buckets, key1="document", key2="doc")

                    # Sacar las atenciones de los documentos #
                    sent_scores = self.selector_model.predict(documents_batch, batch_size=32)

                    # Procesar cada documento #
                    for i, doc_index in enumerate(buckets):
                        n_sents = len(buckets[doc_index]["document"]["sents"])

                        doc_sent_scores = sent_scores[i]

                        # Reajustar el padding para sacar frases #
                        if n_sents < self.max_sents_doc:
                            doc_sent_scores = doc_sent_scores[:n_sents]

                        # Rankear las atenciones y coger las k más atentidas #
                        rank = doc_sent_scores.argsort()[::-1]
                        k_best_sentences = rank[:self.k_max]

                        # Generar los candidatos (combinaciones de las k máximas) #
                        candidate_indices = ut.get_combinations(k_best_sentences)
                        doc_sents = np.array(buckets[doc_index]["document"]["sents"])

                        # Leer y acumular en buckets hasta que la longitud de buckets sea max_size
                        # y empezar a generar batches #
                        buckets[doc_index]["combination_indices"] = candidate_indices

                        # Calcular score de los candidatos #
                        rouge_scores = []
                        for j, candidate_index in enumerate(candidate_indices):
                            candidate_index.sort()  # Ordenar por orden de aparición en el documento

                            candidate_sents = doc_sents[candidate_index]

                            (candi_token_ids, candi_positions,
                             candi_segments, candi_masks) = ut.prepare_inputs(candidate_sents,
                                                                              self.tokenizer,
                                                                              self.max_len_sent_summ,
                                                                              self.max_sents_summ)

                            # Fast rouge #
                            rouge_score = self._rouge.get_scores(" ".join(candidate_sents),
                                                                 " ".join(buckets[doc_index]["reference"]["sents"]))

                            rouge_score = (rouge_score[0]["rouge-1"]["f"] +
                                           rouge_score[0]["rouge-2"]["f"] +
                                           rouge_score[0]["rouge-l"]["f"]) / 3.

                            rouge_scores.append(rouge_score)

                            # Añadir candidato al documento doc_index #
                            buckets[doc_index]["candidates"][j] = {"indices": candidate_index,
                                                                   "token_ids": candi_token_ids,
                                                                   "positions": candi_positions,
                                                                   "segments": candi_segments,
                                                                   "masks": candi_masks,
                                                                   "rouge_score": rouge_score}

                        # Una vez calculados todos los candidatos del documento doc_index #

                        # Calcular ranking de los candidatos #
                        rouge_scores = np.array(rouge_scores)
                        sorted_indices = rouge_scores.argsort()[::-1]
                        buckets[doc_index]["ranking"] = sorted_indices

                    # Una vez calculados todos los candidatos de todos los documentos #

                    # Check de muestra erronea #
                    n_check = (2 ** self.k_max) - 1

                    # Calcular el numero de candidatos para el muestreo aleatorio con reemplazamiento sobre el bucket.
                    # Todos los candidatos salen una vez en un par con otro candidato peor rankeado r(c_i)>r(c_j) i<j)
                    n_total_candidates = sum([len(buckets[k]["ranking"]) for k in buckets])

                    # Random sampling del bucket #
                    # Muestreo aleatorio con reemplazamiento sobre el bucket #
                    for _ in range(n_total_candidates):

                        # Seleccionar un documento aleatorio del bucket #
                        doc_index = np.random.choice(list(buckets.keys()))
                        ranking = buckets[doc_index]["ranking"]

                        # Hay un bug que hace que algunos candidatos no se añadan (error al acceder por indice desde ranking) #
                        # Si la muestra es correcta, el numero de muestras en candidatos y ranking debe coincidir y ambos ser
                        # iguales a la suma de coeficientes binomiales (k, r) 1<=r<=k -> esta suma es (2^k)-1.
                        # Si no coinciden, saltamos la muestra
                        if len(ranking) != n_check and len(buckets[doc_index]["candidates"]) != n_check:
                            continue

                        # Seleccionar un candidato generado del documento #
                        # Se elige una posición i sobre el ranking (c_i) y
                        # y otra j>i (c_j) #
                        pos_i = np.random.randint(0, len(ranking) - 1)
                        pos_j = np.random.randint(pos_i + 1, len(ranking))

                        # Indireccion, se cogen las entradas de c_i y c_j, según su posicion en el ranking #
                        index_i = ranking[pos_i]
                        index_j = ranking[pos_j]

                        # Generar c_i  #
                        candi_token_ids = buckets[doc_index]["candidates"][index_i]["token_ids"]
                        candi_positions = buckets[doc_index]["candidates"][index_i]["positions"]
                        candi_segments = buckets[doc_index]["candidates"][index_i]["segments"]
                        candi_masks = buckets[doc_index]["candidates"][index_i]["masks"]

                        # Generar c_j #
                        candj_token_ids = buckets[doc_index]["candidates"][index_j]["token_ids"]
                        candj_positions = buckets[doc_index]["candidates"][index_j]["positions"]
                        candj_segments = buckets[doc_index]["candidates"][index_j]["segments"]
                        candj_masks = buckets[doc_index]["candidates"][index_j]["masks"]

                        # Generar documento #
                        doc_token_ids = buckets[doc_index]["document"]["token_ids"]
                        doc_positions = buckets[doc_index]["document"]["positions"]
                        doc_segments = buckets[doc_index]["document"]["segments"]
                        doc_masks = buckets[doc_index]["document"]["masks"]

                        # Generar referencia #
                        positive_token_ids = buckets[doc_index]["reference"]["token_ids"]
                        positive_positions = buckets[doc_index]["reference"]["positions"]
                        positive_segments = buckets[doc_index]["reference"]["segments"]
                        positive_masks = buckets[doc_index]["reference"]["masks"]

                        # Add token ids to batch #
                        self.batch_doc_token_ids[self.batch_samples] = doc_token_ids
                        self.batch_positive_token_ids[self.batch_samples] = positive_token_ids
                        self.batch_candi_token_ids[self.batch_samples] = candi_token_ids
                        self.batch_candj_token_ids[self.batch_samples] = candj_token_ids

                        # Add positions to batch #
                        self.batch_doc_positions[self.batch_samples] = doc_positions
                        self.batch_positive_positions[self.batch_samples] = positive_positions
                        self.batch_candi_positions[self.batch_samples] = candi_positions
                        self.batch_candj_positions[self.batch_samples] = candj_positions

                        # Add segments to batch #
                        self.batch_doc_segments[self.batch_samples] = doc_segments
                        self.batch_positive_segments[self.batch_samples] = positive_segments
                        self.batch_candi_segments[self.batch_samples] = candi_segments
                        self.batch_candj_segments[self.batch_samples] = candj_segments

                        # Add masks to batch #
                        self.batch_doc_masks[self.batch_samples] = doc_masks
                        self.batch_positive_masks[self.batch_samples] = positive_masks
                        self.batch_candi_masks[self.batch_samples] = candi_masks
                        self.batch_candj_masks[self.batch_samples] = candj_masks

                        # Add difference in ranking #
                        self.batch_diffs_ranking[self.batch_samples] = pos_j - pos_i

                        self.batch_samples += 1

                        # Si el numero de muestras en el batch == batch_size devolver el batch
                        if self.batch_samples == self.batch_size:
                            self.batch_samples = 0

                            if self.train:
                                yield ({"doc_token_ids": self.batch_doc_token_ids,
                                        "doc_positions": self.batch_doc_positions,
                                        "doc_segments": self.batch_doc_segments,
                                        "doc_masks": self.batch_doc_masks,
                                        "pos_token_ids": self.batch_positive_token_ids,
                                        "pos_positions": self.batch_positive_positions,
                                        "pos_segments": self.batch_positive_segments,
                                        "pos_masks": self.batch_positive_masks,
                                        "candi_token_ids": self.batch_candi_token_ids,
                                        "candi_positions": self.batch_candi_positions,
                                        "candi_segments": self.batch_candi_segments,
                                        "candi_masks": self.batch_candi_masks,
                                        "candj_token_ids": self.batch_candj_token_ids,
                                        "candj_positions": self.batch_candj_positions,
                                        "candj_segments": self.batch_candj_segments,
                                        "candj_masks": self.batch_candj_masks},
                                       {"outputs": np.zeros(self.batch_size, dtype="int32"),
                                        "outputs2": self.batch_diffs_ranking})
                            else:
                                yield {"doc_token_ids": self.batch_doc_token_ids,
                                       "doc_positions": self.batch_doc_positions,
                                       "doc_segments": self.batch_doc_segments,
                                       "doc_masks": self.batch_doc_masks,
                                       "pos_token_ids": self.batch_positive_token_ids,
                                       "pos_positions": self.batch_positive_positions,
                                       "pos_segments": self.batch_positive_segments,
                                       "pos_masks": self.batch_positive_masks,
                                       "candi_token_ids": self.batch_candi_token_ids,
                                       "candi_positions": self.batch_candi_positions,
                                       "candi_segments": self.batch_candi_segments,
                                       "candi_masks": self.batch_candi_masks}

                    # Cuando finaliza el random sampling se reinicia el bucket #
                    buckets = {}

class SummarizeMatchGenerator:

    def __init__(self, dataset_file, tokenizer,
                 aes_select, batch_size, max_len_sent_doc,
                 max_sents_doc, max_len_sent_summ,
                 max_sents_summ, k_max, ngram_blocking, sent_split="<s>"):

        self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len_sent_doc = max_len_sent_doc
        self.max_sents_doc = max_sents_doc
        self.max_len_sent_summ = max_len_sent_summ
        self.max_sents_summ = max_sents_summ
        self.k_max = k_max
        self.ngram_blocking = ngram_blocking
        self.sent_split = sent_split
        self.selector_model = aes_select.selector_model
        self.max_len_seq_doc = (self.max_len_sent_doc * self.max_sents_doc) + (self.max_sents_doc * 2)
        self.max_len_seq_summ = (self.max_len_sent_summ * self.max_sents_summ) + (self.max_sents_summ * 2)

        self.batch_doc_token_ids = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_doc_positions = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_doc_segments = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")
        self.batch_doc_masks = np.zeros((self.batch_size, self.max_len_seq_doc), dtype="int32")

        self.batch_candi_token_ids = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_candi_positions = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_candi_segments = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")
        self.batch_candi_masks = np.zeros((self.batch_size, self.max_len_seq_summ), dtype="int32")

        self.bucket_max_size = 5
        self.batch_samples = 0


    def generator(self):
        fr = open(self.dataset_file, "r", encoding="utf8")
        fr.readline() # Skip header
        lines = fr.readlines()
        buckets = {}

        for doc_index, line in enumerate(lines):
            # Separar documento y referencia #
            doc, summ = line.split("\t")
            doc_sents = ut.preprocess_text(doc, self.sent_split)
            summ_sents = ut.preprocess_text(summ, self.sent_split)

            # Comprobar que documento y referencia son validos #
            if not doc_sents or not summ_sents:
                continue

            # Preparo documento #
            (doc_token_ids, doc_positions,
             doc_segments, doc_masks) = ut.prepare_inputs(doc_sents,
                                                          self.tokenizer,
                                                          self.max_len_sent_doc,
                                                          self.max_sents_doc)

            # Añadir documento al bucket #
            buckets[doc_index] = {"document": {"token_ids": doc_token_ids,
                                               "positions": doc_positions,
                                               "segments": doc_segments,
                                               "masks": doc_masks,
                                               "sents": doc_sents},
                                  "reference": {"sents": summ_sents},
                                  "candidates": {},
                                  "combination_indices": [],
                                  }

            # Si se ha llenado el bucket, ya tenemos todos los documentos y referencias #
            if len(buckets) == self.bucket_max_size:
                # Batchify de los documentos del bucket #
                documents_batch = ut.get_batch(buckets, key1="document", key2="doc")

                # Sacar las atenciones de los documentos #
                sent_scores = self.selector_model.predict(documents_batch, batch_size=32)

                # Procesar cada documento #
                for i, doc_index in enumerate(buckets):
                    n_sents = len(buckets[doc_index]["document"]["sents"])

                    doc_sent_scores = sent_scores[i]

                    # Reajustar el padding para sacar frases #
                    if n_sents < self.max_sents_doc:
                        doc_sent_scores = doc_sent_scores[:n_sents]

                    # Rankear las atenciones y coger las k más atentidas #
                    rank = doc_sent_scores.argsort()[::-1]
                    k_best_sentences = rank[:self.k_max]

                    # Generar los candidatos (combinaciones de las k máximas) #
                    candidate_indices = ut.get_combinations(k_best_sentences)
                    doc_sents = np.array(buckets[doc_index]["document"]["sents"])
                    buckets[doc_index]["combination_indices"] = candidate_indices

                    # Generar los candidatos #
                    for j, candidate_index in enumerate(candidate_indices):
                        candidate_index.sort()  # Ordenar por orden de aparición en el documento
                        candidate_sents = doc_sents[candidate_index]

                        (candi_token_ids, candi_positions,
                         candi_segments, candi_masks) = ut.prepare_inputs(candidate_sents,
                                                                          self.tokenizer,
                                                                          self.max_len_sent_summ,
                                                                          self.max_sents_summ)

                        buckets[doc_index]["candidates"][j] = {"indices": candidate_index,
                                                               "token_ids": candi_token_ids,
                                                               "positions": candi_positions,
                                                               "segments": candi_segments,
                                                               "masks": candi_masks,
                                                               "sents": candidate_sents}

                    # Una vez calculados los candidatos para el documento #
                    # Se devuelve toda la información de ese documento (incluyendo referencia y candidatos) #
                    yield buckets[doc_index]

                # Una vez procesados todos los documentos del bucket, reiniciarlo #
                buckets = {}