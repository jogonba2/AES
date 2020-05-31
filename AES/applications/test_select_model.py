from AES.models.select_model import SelectModel
from AES.utils.generators import SelectGenerator
from AES.optimization import metrics as metric_funcs
from AES.utils import visualization
import transformers
import json
import sys
import subprocess
import numpy as np


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

    # Params for optimization process #
    loss = config["optimization"]["loss"]
    margin = config["optimization"]["margin"]
    learning_rate = config["optimization"]["learning_rate"]
    metrics = config["optimization"]["metrics"]
    ranking_triplet_func = config["optimization"]["ranking_triplet_func"]

    # Params for test process #
    batch_size = config["inference_params"]["batch_size"]
    path_load_weights = config["inference_params"]["path_load_weights"]
    avg_att_layers = config["inference_params"]["avg_att_layers"]
    verbose = config["inference_params"]["verbose"]

    # Create model #
    aes = SelectModel(model_name=model_name,
                   shortcut_weights=shortcut_weights,
                   max_len_sent_doc=max_len_sent_doc,
                   max_sents_doc=max_sents_doc,
                   max_len_sent_summ=max_len_sent_summ,
                   max_sents_summ=max_sents_summ,
                   learning_rate=learning_rate,
                   loss=loss,
                   margin=margin,
                   metrics=metrics,
                   avg_att_layers=avg_att_layers,
                   train=False)
    aes.build()
    aes.compile()
    aes.model.load_weights(path_load_weights)


    test_generator = SelectGenerator(dataset_file, tokenizer,
                                       aes.triplet_mining_model, batch_size,
                                       max_len_sent_doc, max_sents_doc,
                                       max_len_sent_summ, max_sents_summ,
                                       sent_split="<s>", train=False,
                                       ranking_triplet_func=ranking_triplet_func).generator

    n_dataset = int(subprocess.run(['wc', '-l', dataset_file],
                    stdout=subprocess.PIPE).stdout.split()[0].strip()) - 1

    for batch in test_generator():
        preds = aes.model.predict(batch)
        alphas = aes.selector_model.predict({"doc_token_ids": batch["doc_token_ids"],
                                             "doc_positions": batch["doc_positions"],
                                             "doc_segments": batch["doc_segments"],
                                             "doc_masks": batch["doc_masks"]})


        print("Similarity Cand-Pos:", metric_funcs.cosine_similarity_pos_pair(None,
                                                                              np.expand_dims(preds[0], 0)).numpy())
        print("Similarity Cand-Neg:", metric_funcs.cosine_similarity_neg_pair(None,
                                                                              np.expand_dims(preds[0], 0)).numpy())

        print(alphas[0].shape)
        first_head = alphas[0]
        print(alphas[0])
        #print("Suma por filas", first_head.sum(axis=0))
        #print("Suma por columnas", first_head.sum(axis=1))
        visualization.att_visualization(np.expand_dims(alphas[0], 0))