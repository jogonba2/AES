from AES.models.select_model import SelectModel
from AES.utils.generators import SummarizeSelectGenerator
from AES.utils import summarization_utils
from pythonrouge.pythonrouge import Pythonrouge
import transformers
import json
import sys
import subprocess

# [3, 4, 5, 6, 7, 8, 9, 11, 2, 0] #

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

    # Params for summarization process #
    batch_size = config["inference_params"]["batch_size"]
    path_load_weights = config["inference_params"]["path_load_weights"]
    verbose = config["inference_params"]["verbose"]
    k = config["inference_params"]["k"]
    avg_att_layers = config["inference_params"]["avg_att_layers"]
    ngram_blocking = config["inference_params"]["ngram_blocking"]

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


    test_generator = SummarizeSelectGenerator(dataset_file, tokenizer,
                                       batch_size,
                                       max_len_sent_doc, max_sents_doc,
                                       sent_split="<s>").generator

    n_dataset = int(subprocess.run(['wc', '-l', dataset_file],
                    stdout=subprocess.PIPE).stdout.split()[0].strip()) - 1

    hyps = []
    refs = []
    leads = []
    c = 0
    for (batch, (doc_sents, summ_sents)) in test_generator():
        alphas = aes.selector_model.predict({"doc_token_ids": batch["doc_token_ids"],
                                             "doc_positions": batch["doc_positions"],
                                             "doc_segments": batch["doc_segments"],
                                             "doc_masks": batch["doc_masks"]})

        if c == 15: break
        for i in range(batch_size):

            scores = alphas[i]

            if ngram_blocking:
                gen_summary = summarization_utils.selection_summary_ngram_blocking(scores, doc_sents[i],
                                                                                   max_sents_doc, k,
                                                                                   block_ngrams=ngram_blocking)
            else:
                gen_summary = summarization_utils.selection_summary(scores, doc_sents[i],
                                                                    max_sents_doc, k)

            if verbose:
                for j, sent in enumerate(doc_sents[i]):
                    print(j,") ", sent)

                print("-"*20)

                for j, sent in enumerate(summ_sents[i]):
                    print(j,") ", sent)

                print("-"*20)
                print(scores)
                for j, sent in enumerate(gen_summary):
                    print(j,") ", sent)
                input()

            ref_summary = summ_sents[i]
            hyps.append([s.strip() for s in gen_summary])
            refs.append([[s.strip() for s in ref_summary]])
            leads.append([s.strip() for s in doc_sents[i][:k]])

        c += 1
        if c % 10 == 0:
            print(c)

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
