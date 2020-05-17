import tensorflow as tf
import AES.optimization.losses as losses_module
import numpy as np
from AES.optimization import lr_annealing
from AES.optimization import metrics as metric_funcs


def fit(model, train_dataset, n_dataset, epochs, batch_size, callbacks):
    model.fit(train_dataset, epochs=epochs,
              steps_per_epoch=n_dataset // batch_size,
              callbacks=callbacks)

def grad_accum_fit(model, train_dataset, epochs,
                   match_loss, margin,
                   select_loss, grad_accum_iters=1,
                   noam_annealing=False, noam_params=None,
                   hour_checkpointing=True,
                   loss_weights={"match": 1,
                                 "select": 1}):

    if noam_annealing:
        noam_schedule = lr_annealing.Noam(noam_params["warmup_steps"],
                                          noam_params["hidden_dims"])
        optimizer = tf.keras.optimizers.Adam(learning_rate=noam_schedule)
    else:
        optimizer = tf.keras.optimizers.Adam()
    match_loss = getattr(losses_module, match_loss)(margin)
    select_loss = getattr(losses_module, select_loss)
    loss_history = np.zeros((1, 3), "float32")
    metrics = np.zeros((1, 3), "float32")

    for (n_batch, (inputs, _)) in enumerate(train_dataset):
        with tf.GradientTape(persistent=True) as tape:
            if n_batch % grad_accum_iters == 0:
                if n_batch > 0:
                    optimizer.apply_gradients(zip(accum_grads, tvs))
                tvs = model.trainable_variables
                accum_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]

            logits = model(inputs, training=True)
            match_loss_value = match_loss(None, logits[0])
            select_loss_value = select_loss(None, logits[1])
            total_loss = loss_weights["match"] * match_loss_value + loss_weights["select"] * select_loss_value
            doc_summ_sim = tf.keras.backend.mean(metric_funcs.cos_sim_doc_summ(None, logits[0]))
            doc_cand_sim = tf.keras.backend.mean(metric_funcs.cos_sim_doc_cand(None, logits[0]))
            summ_cand_sim = tf.keras.backend.mean(metric_funcs.cos_sim_summ_cand(None, logits[1]))

        loss_history = np.vstack((loss_history, np.array([total_loss.numpy().mean(),
                                                         match_loss_value.numpy().mean(),
                                                         select_loss_value.numpy().mean()])))

        metrics = np.vstack((metrics, np.array([doc_summ_sim,
                                               doc_cand_sim,
                                               summ_cand_sim])))

        # Gradients for each loss #
        grads = tape.gradient(total_loss, tvs)

        # Update gradients for both losses #
        for i, grad in enumerate(grads):
            if grad is not None:
                accum_grads[i].assign_add(grad)

        # Monitorize losses and metrics #
        print("(Batch: %d) Total loss: %.4f: Matching loss: %.4f, Selecting loss: %.4f  |||" \
              " Doc-Summ dist: %.4f, Doc-Cand dist: %.4f, Cand-Summ dist: %.4f"%
              (n_batch, loss_history[1:, 0].mean(), loss_history[1:, 1].mean(), loss_history[1:, 2].mean(),
               metrics[1:, 0].mean(), metrics[1:, 1].mean(), metrics[1:, 2].mean()), end="\r")