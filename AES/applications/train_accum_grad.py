from AES.models.aes import AESModel
from AES.utils.generator import TrainGenerator
from AES.optimization.lr_annealing import Noam
from AES.optimization.train_schedule import fit, grad_accum_fit
from AES.utils.callbacks import TimeCheckpoint
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
    tokenizer = getattr(transformers,
                        tokenizer_name).from_pretrained(shortcut_weights)

    # Params for AES model #
    att_layer = config["model"]["att_layer"]
    att_params = config["model"]["att_params"]
    att_params["max_sents"] = max_sents_doc

    # Params for optimization process #
    match_loss = config["optimization"]["match_loss"]
    margin = config["optimization"]["margin"]
    select_loss = config["optimization"]["select_loss"]
    loss_weights = config["optimization"]["loss_weights"]
    noam_annealing = config["optimization"]["noam_annealing"]
    if noam_annealing:
        warmup_steps = config["optimization"]["warmup_steps"]
    noam_params = None
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
                   match_loss=match_loss,
                   margin=margin,
                   select_loss=select_loss,
                   metrics=metrics)
    aes.build()
    aes.compile()
    print(aes.model.summary())

    tr_generator = TrainGenerator(dataset_file, tokenizer,
                                  max_len_sent_doc, max_sents_doc,
                                  max_len_sent_summ, max_sents_summ,
                                  sent_split=sent_split).generator

    # Create Dataset from generator #
    dataset = tf.data.Dataset.from_generator(tr_generator,
                                             ({'doc_token_ids': tf.int32,
                                               'doc_positions': tf.int32,
                                               'doc_segments': tf.int32,
                                               'doc_masks': tf.int32,
                                               'summ_token_ids': tf.int32,
                                               'summ_positions': tf.int32,
                                               'summ_segments': tf.int32,
                                               'summ_masks': tf.int32,
                                               }, {"matching": tf.int32,
                                                   "selecting": tf.int32}))

    n_dataset = int(subprocess.run(['wc', '-l', dataset_file],
                    stdout=subprocess.PIPE).stdout.split()[0].strip()) - 1
    train_dataset = dataset.shuffle(512).batch(batch_size).repeat(-1)

    # Annealing params #
    if noam_annealing:
        noam_params = {"warmup_steps": warmup_steps,
                       "hidden_dims": hidden_dim}

    #fit(aes.model, train_dataset, 500, 2, 8, None) # Este funciona perfecto!
    aes.save_model(path_save_weights)
    """
    # Custom training for accumulating gradient #
    grad_accum_fit(aes.model, train_dataset, epochs,
                   match_loss, margin, select_loss,
                   grad_accum_iters=grad_accum_iters,
                   noam_annealing=noam_annealing,
                   noam_params=noam_params,
                   hour_checkpointing=True,
                   loss_weights=loss_weights)
    """
    #aes.save_model(path_save_weights)



