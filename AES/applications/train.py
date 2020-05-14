from AES.models.aes import AESModel
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
    max_len_sent = config["representation"]["max_len_sent"]
    max_sents = config["representation"]["max_sents"]

    # Model params for HuggingFace Transformer implementation #
    model_name = config["huggingface"]["model_name"]
    shortcut_weights = config["huggingface"]["shortcut_weights"]
    tokenizer_name = config["huggingface"]["tokenizer"]
    tokenizer = getattr(transformers,
                        tokenizer_name).from_pretrained(shortcut_weights)

    # Params for the loss functions #
    alpha = config["loss_params"]["alpha"]
    gamma = config["loss_params"]["gamma"]

    # Params for optimization process #
    optimizer = config["optimization"]["optimizer"]
    noam_annealing = config["optimization"]["noam_annealing"]
    if noam_annealing:
        warmup_steps = config["optimization"]["warmup_steps"]
    grad_accum_iters = config["optimization"]["grad_accum_iters"]

    # Params for training process #
    batch_size = config["training_params"]["batch_size"]
    epochs = config["training_params"]["epochs"]
    path_save_weights = config["training_params"]["path_save_weights"]
    verbose = config["training_params"]["verbose"]

    # Create model #
    aes = AESModel(model_name, shortcut_weights,
                   max_len_sent, max_sents, optimizer)
    aes.build()
    aes.compile()
    print(aes.tr_model.summary())
    print(aes.tr_model.output_shape)
