from AES.models.select_model import SelectModel
from AES.models.match_model import MatchModel
from AES.utils.generators import MatchGenerator
from AES.optimization.train_schedule import grad_accum_match_fit
from AES.utils import callbacks
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

    # Params for optimization process (match model) #
    loss_1 = config["optimization"]["loss_1"]
    loss_2 = config["optimization"]["loss_2"]
    margin_1 = config["optimization"]["margin_1"]
    margin_2 = config["optimization"]["margin_2"]
    k_max = config["optimization"]["k_max"]
    avg_att_layers = config["optimization"]["avg_att_layers"]
    ngram_blocking = config["optimization"]["ngram_blocking"]
    learning_rate = config["optimization"]["learning_rate"]
    noam_annealing = config["optimization"]["noam_annealing"]
    grad_accum_iters = config["optimization"]["grad_accum_iters"]
    metrics = config["optimization"]["metrics"]
    if noam_annealing:
        noam_params = {"warmup_steps": config["optimization"]["warmup_steps"],
                       "hidden_dims": hidden_dim}
    else:
        noam_params = None

    # Params for training process #
    batch_size = config["training_params"]["batch_size"]
    epochs = config["training_params"]["epochs"]
    path_save_weights = config["training_params"]["path_save_weights"]
    verbose = config["training_params"]["verbose"]

    # Definir aes_full. El modelo es idéntico a match_model, pero el
    # extractor está basado en nuestros modelos siamese attentional.
    # Se usan las atenciones del encoder a nivel de frases para extraer resúmenes
    # durante el entrenamiento del modelo completo. #
    aes_full = MatchModel(model_name, shortcut_weights,
                          max_len_sent_doc, max_sents_doc,
                          max_len_sent_summ, max_sents_summ,
                          noam_annealing=noam_annealing,
                          noam_params=noam_params,
                          learning_rate=learning_rate,
                          loss_1=loss_1, loss_2=loss_2,
                          margin_1=margin_1, margin_2=margin_2,
                          metrics=metrics, avg_att_layers=avg_att_layers,
                          train=True)
    aes_full.build()
    aes_full.compile()

    n_dataset = int(subprocess.run(['wc', '-l', dataset_file],
                    stdout=subprocess.PIPE).stdout.split()[0].strip()) - 1

    train_generator = MatchGenerator(dataset_file, tokenizer,
                                     aes_full, k_max, avg_att_layers,
                                     ngram_blocking, batch_size, max_len_sent_doc,
                                     max_sents_doc, max_len_sent_summ,
                                     max_sents_summ, sent_split="<s>",
                                     train=True).generator

    if grad_accum_iters > 1:
        grad_accum_match_fit(aes_full, train_generator(), epochs,
                              (n_dataset // batch_size) * 10,
                              optimizer=aes_full.get_optimizer(),
                              path_save_weights=path_save_weights,
                              grad_accum_iters=grad_accum_iters,
                              hour_checkpointing=True)

    else:
        hour_checkpointing = callbacks.TimeCheckpoint(hours_step=1,
                                                      path=path_save_weights,
                                                      aes=aes_full)
        aes_full.model.fit(train_generator(),
                      steps_per_epoch=(n_dataset // batch_size)*10,
                      epochs=1, callbacks=[hour_checkpointing])


    aes_full.save_model(aes_full.model, path_save_weights)
