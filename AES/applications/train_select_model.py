from AES.models.select_model import SelectModel
from AES.utils.generators import SelectGenerator
from AES.optimization.train_schedule import grad_accum_select_fit
from AES.utils import callbacks
import tensorflow as tf
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

    # Params for optimization process #
    loss = config["optimization"]["loss"]
    margin = config["optimization"]["margin"]
    learning_rate = config["optimization"]["learning_rate"]
    noam_annealing = config["optimization"]["noam_annealing"]
    grad_accum_iters = config["optimization"]["grad_accum_iters"]
    metrics = config["optimization"]["metrics"]
    ranking_triplet_func = config["optimization"]["ranking_triplet_func"]
    if noam_annealing:
        noam_params = {"warmup_steps": config["optimization"]["warmup_steps"],
                       "hidden_dims": hidden_dim}
    else:
        noam_params = None
    # Annealing params #


    # Params for training process #
    batch_size = config["training_params"]["batch_size"]
    epochs = config["training_params"]["epochs"]
    path_save_weights = config["training_params"]["path_save_weights"]
    verbose = config["training_params"]["verbose"]



    # Create model #
    aes = SelectModel(model_name=model_name,
                   shortcut_weights=shortcut_weights,
                   max_len_sent_doc=max_len_sent_doc,
                   max_sents_doc=max_sents_doc,
                   max_len_sent_summ=max_len_sent_summ,
                   max_sents_summ=max_sents_summ,
                   noam_annealing=noam_annealing,
                   noam_params=noam_params,
                   learning_rate=learning_rate,
                   loss=loss,
                   margin=margin,
                   metrics=metrics)

    aes.build()
    aes.compile()

    train_generator = SelectGenerator(dataset_file, tokenizer,
                                       aes.triplet_mining_model, batch_size,
                                       max_len_sent_doc, max_sents_doc,
                                       max_len_sent_summ, max_sents_summ,
                                       sent_split="<s>",
                                       ranking_triplet_func=ranking_triplet_func).generator

    n_dataset = int(subprocess.run(['wc', '-l', dataset_file],
                    stdout=subprocess.PIPE).stdout.split()[0].strip()) - 1


    if grad_accum_iters > 1:
        grad_accum_select_fit(aes, train_generator(), epochs,
                              loss, margin, n_dataset // batch_size,
                              optimizer=aes.get_optimizer(),
                              path_save_weights=path_save_weights,
                              grad_accum_iters=grad_accum_iters,
                              hour_checkpointing=True)

    else:
        hour_checkpointing = callbacks.TimeCheckpoint(hours_step=1,
                                                      path=path_save_weights,
                                                      aes=aes)
        aes.model.fit(train_generator(),
                      steps_per_epoch=n_dataset // batch_size,
                      epochs=1, callbacks=[hour_checkpointing])


    aes.save_model(aes.model, path_save_weights)
