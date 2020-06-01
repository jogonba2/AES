from AES.models.match_model import MatchModel
from AES.models.select_model import SelectModel
from AES.utils.generators import SummarizeMatchGenerator
from AES.utils import summarization_utils
from AES.utils import utils
from pythonrouge.pythonrouge import Pythonrouge
import tensorflow as tf
import numpy as np
import transformers
import json
import sys
import subprocess


if __name__ == "__main__":
    # Load config file #
    with open(sys.argv[1], "r") as json_file:
        config = json.load(json_file)

    # Load dataset #
    dataset_file = config["dataset"]["file"]

    # Representation params #
    max_len_sent_doc = config["representation"]["max_len_sent_doc"]
    max_sents_doc = config["representation"]["max_sents_doc"]
    max_len_sent_summ = config["representation"]["max_len_sent_summ"]
    max_sents_summ = config["representation"]["max_sents_summ"]
    sent_split = config["representation"]["sent_split"]

    # Params for HuggingFace Transformer implementation #
    model_name = config["huggingface"]["model_name"]
    shortcut_weights = config["huggingface"]["shortcut_weights"]
    tokenizer_name = config["huggingface"]["tokenizer"]
    hidden_dim = config["huggingface"]["output_dim"]
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(shortcut_weights)

    # Params of match model #
    loss_1 = config["match_params"]["loss_1"]
    loss_2 = config["match_params"]["loss_2"]
    margin_1 = config["match_params"]["margin_1"]
    margin_2 = config["match_params"]["margin_2"]
    learning_rate = config["match_params"]["learning_rate"]
    metrics = config["match_params"]["metrics"]

    # Params of select model #
    select_loss = config["select_params"]["loss"]
    select_margin = config["select_params"]["margin"]

    # Params for summarization process #
    batch_size = config["inference_params"]["batch_size"]
    path_match_weights = config["inference_params"]["path_match_weights"]
    path_select_weights = config["inference_params"]["path_select_weights"]
    verbose = config["inference_params"]["verbose"]
    k_max = config["inference_params"]["k_max"]
    avg_att_layers = config["inference_params"]["avg_att_layers"]
    ngram_blocking = config["inference_params"]["ngram_blocking"]


    # Create model aes_select #
    aes_select = SelectModel(model_name=model_name,
                              shortcut_weights=shortcut_weights,
                              max_len_sent_doc=max_len_sent_doc,
                              max_sents_doc=max_sents_doc,
                              max_len_sent_summ=max_len_sent_summ,
                              max_sents_summ=max_sents_summ,
                              learning_rate=learning_rate,
                              loss=select_loss,
                              margin=select_margin,
                              metrics=metrics,
                              avg_att_layers=avg_att_layers,
                              train=False)
    aes_select.build()
    aes_select.compile()
    aes_select.model.load_weights(path_select_weights)

    # Create match model #
    aes_match = MatchModel(model_name, shortcut_weights,
                             max_len_sent_doc, max_sents_doc,
                             max_len_sent_summ, max_sents_summ,
                             learning_rate=learning_rate,
                             loss_1=loss_1,
                             loss_2=loss_2,
                             margin_1=margin_1,
                             margin_2=margin_2,
                             metrics=metrics,
                             avg_att_layers=avg_att_layers,
                             train=True)
    aes_match.build()
    aes_match.compile()
    aes_match.model.load_weights(path_match_weights)


    test_generator = SummarizeMatchGenerator(dataset_file, tokenizer,
                                             aes_select, batch_size,
                                             max_len_sent_doc, max_sents_doc,
                                             max_len_sent_summ, max_sents_summ,
                                             k_max, ngram_blocking, sent_split="<s>").generator

    n_dataset = int(subprocess.run(['wc', '-l', dataset_file],
                    stdout=subprocess.PIPE).stdout.split()[0].strip()) - 1

    hyps = []
    refs = []
    leads = []
    c = 0
    for doc_bucket in test_generator():
        # Batchify de candidatos #
        cands_batch = utils.get_candidates_batch(doc_bucket["candidates"], "candi")

        # Batch de un documento #
        doc_batch = {"doc_token_ids": np.array([doc_bucket["document"]["token_ids"]]),
                     "doc_positions": np.array([doc_bucket["document"]["positions"]]),
                     "doc_segments": np.array([doc_bucket["document"]["segments"]]),
                     "doc_masks": np.array([doc_bucket["document"]["masks"]])}


        # Calcular representacion de documento #
        doc_repr = aes_match.doc_repr_model.predict(doc_batch)[0]

        # Calcular representaciones de los candidatos #
        cands_repr = aes_match.candi_repr_model.predict(cands_batch, batch_size=32)
        best_cand_ind = -1
        best_sim = float("-inf")
        for i in range(len(cands_repr)):
            cand_repr = cands_repr[i]
            cos_sim = -tf.keras.losses.cosine_similarity(doc_repr, cand_repr, axis=-1).numpy()

            # Si no es un candidato valido, asignarle cos_sim negativa infinita #
            if utils.check_ngram_blocking(doc_bucket["candidates"][i]["sents"], ngram_blocking):
                cos_sim = float("-inf")
            print(i, cos_sim)
            if cos_sim > best_sim:
                best_cand_ind = i
                best_sim = cos_sim

        hyp = doc_bucket["candidates"][best_cand_ind]["sents"]
        ref = doc_bucket["reference"]["sents"]
        hyps.append([s.strip() for s in hyp])
        leads.append([s.strip() for s in doc_bucket["document"]["sents"][:3]])
        refs.append([[s.strip() for s in ref]])

        c += 1
        if c % 30 == 0 and c > 0:
            print(c)
            break

    rouge = Pythonrouge(summary_file_exist=False, delete_xml=True,
                        summary=hyps, reference=refs,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                        f_measure_only=False, stemming=True, stopwords=False,
                        word_level=True, length_limit=False)

    print("AES:", rouge.calc_score())

    rouge = Pythonrouge(summary_file_exist=False, delete_xml=True,
                        summary=leads, reference=refs,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                        f_measure_only=False, stemming=True, stopwords=False,
                        word_level=True, length_limit=False)

    print("LEAD:", rouge.calc_score())