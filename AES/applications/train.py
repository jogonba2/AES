from AES.models.aes import AESModel
from AES.utils.generator import TrainGenerator
import transformers
import json
import sys


if __name__ == "__main__":
    # Load config file #
    with open(sys.argv[1], "r") as json_file:
        config = json.load(json_file)

    # Load dataset #
    dataset = config["dataset"]["file"]

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
    tokenizer = getattr(transformers,
                        tokenizer_name).from_pretrained(shortcut_weights)

    # Params for AES model #
    att_layer = config["model"]["att_layer"]
    att_params = config["model"]["att_params"]
    att_params["max_sents"] = max_sents_doc

    # Params for optimization process #
    loss = config["optimization"]["loss"]
    margin = config["optimization"]["margin"]
    optimizer = config["optimization"]["optimizer"]
    noam_annealing = config["optimization"]["noam_annealing"]
    if noam_annealing:
        warmup_steps = config["optimization"]["warmup_steps"]
    grad_accum_iters = config["optimization"]["grad_accum_iters"]
    metrics = config["optimization"]["metrics"]

    # Params for training process #
    batch_size = config["training_params"]["batch_size"]
    epochs = config["training_params"]["epochs"]
    path_save_weights = config["training_params"]["path_save_weights"]
    verbose = config["training_params"]["verbose"]

    # Create model #

    aes = AESModel(model_name=model_name,
                   shortcut_weights=shortcut_weights,
                   max_len_sent_doc=max_len_sent_doc,
                   max_sents_doc=max_sents_doc,
                   max_len_sent_summ=max_len_sent_summ,
                   max_sents_summ=max_sents_summ,
                   att_layer=att_layer,
                   att_params=att_params,
                   optimizer=optimizer,
                   loss=loss,
                   margin=margin,
                   metrics=metrics)
    aes.build()
    aes.compile()
    print(aes.tr_model.summary())

    tr_generator = TrainGenerator(dataset, batch_size,
                                  tokenizer, max_len_sent_doc,
                                  max_sents_doc, max_len_sent_summ,
                                  max_sents_summ, sent_split=sent_split).generator()
    aes.tr_model.fit_generator(tr_generator, steps_per_epoch=100, epochs=2)

    print(next(tr_generator))

