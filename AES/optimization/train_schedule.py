import tensorflow as tf
import AES.optimization.losses as losses_module
from AES.utils import utils
from datetime import datetime
from AES.optimization import metrics as metric_funcs
from tqdm import tqdm

def fit(model, train_dataset, steps_per_epoch, epochs, callbacks):
    model.fit(train_dataset, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)

@tf.function
def compute_total_loss(model, inputs, loss, training=True):
    logits = model(inputs, training=training)
    loss_value = loss(None, logits)
    return logits, loss_value

@tf.function
def get_grad(model, inputs, loss, training=True):
    with tf.GradientTape() as tape:
        logits, loss_value = compute_total_loss(model, inputs, loss, training=training)
    return logits, loss_value, tape.gradient(loss_value, model.trainable_variables)

def grad_accum_select_fit(aes, train_generator, epochs,
                   loss, margin, batches_per_epoch,
                   optimizer, path_save_weights,
                   grad_accum_iters=1, hour_checkpointing=True):


    if hour_checkpointing:
        prev_time = datetime.utcnow()

    model = aes.model
    loss = getattr(losses_module, loss)(margin)
    tvs = model.trainable_variables
    accum_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
    train_loss_results = []

    for e in range(epochs):
        total_loss_avg = tf.keras.metrics.Mean()
        sim_cand_pos_avg = tf.keras.metrics.Mean()
        sim_cand_neg_avg = tf.keras.metrics.Mean()

        tqbar = tqdm(enumerate(train_generator), total=batches_per_epoch)

        for (n_batch, (inputs, _)) in tqbar:
            logits, loss_value, grads = get_grad(model, inputs, loss, training=True)

            for i, grad in enumerate(grads):
                if grad is not None:
                    accum_grads[i].assign_add(grad).__div__(grad_accum_iters)

            if n_batch % grad_accum_iters == 0:
                if n_batch > 0:
                    optimizer.apply_gradients(zip(accum_grads, model.trainable_variables))
                accum_grads = [var.assign(tf.zeros_like(var.initialized_value())) for var in accum_grads]


            sim_cand_pos = metric_funcs.cosine_similarity_pos_pair(None, logits)
            sim_cand_neg = metric_funcs.cosine_similarity_neg_pair(None, logits)

            total_loss_avg.update_state(loss_value)

            sim_cand_pos_avg.update_state(sim_cand_pos)
            sim_cand_neg_avg.update_state(sim_cand_neg)

            # Monitorize losses and metrics after the batch #
            tqbar.set_description("Total loss: %.4f: |||" \
                  " Cand-Pos sim: %.4f, Cand-Neg sim: %.4f\n"%
                  (total_loss_avg.result(), sim_cand_pos_avg.result(), sim_cand_neg_avg.result()))
            tqbar.refresh()

            if hour_checkpointing:
                save_flag = utils.diff_hours(prev_time)
                if save_flag:
                    prev_time = datetime.utcnow()
                    aes.save_model(model, path_save_weights)

        train_loss_results.append(total_loss_avg.result())

        # Monitorize losses and metrics after the epoch #
        tqbar.set_description("Total loss: %.4f: |||" \
                              " Cand-Pos sim: %.4f, Cand-Neg sim: %.4f\n" %
                              (total_loss_avg.result(), sim_cand_pos_avg.result(), sim_cand_neg_avg.result()))
        tqbar.refresh()



@tf.function
def match_compute_total_loss(model, inputs, loss_1, loss_2, diff_rankings, training=True):
    logits = model(inputs, training=training)
    loss_value_1 = loss_1(None, logits[0])
    loss_value_2 = loss_2(diff_rankings, logits[1])
    return logits, loss_value_1, loss_value_2

@tf.function
def match_get_grad(model, inputs, loss_1, loss_2, diff_rankings, training=True):
    with tf.GradientTape() as tape:
        logits, loss_value_1, loss_value_2 = match_compute_total_loss(model, inputs, loss_1, loss_2,
                                                                     diff_rankings, training=training)
    loss_value = [loss_value_1, loss_value_2]
    return logits, loss_value_1, loss_value_2, tape.gradient(loss_value, model.trainable_variables)

def grad_accum_match_fit(aes, train_generator, epochs,
                               batches_per_epoch,
                               optimizer, path_save_weights,
                               grad_accum_iters=1, hour_checkpointing=True):


    if hour_checkpointing:
        prev_time = datetime.utcnow()

    model = aes.model
    loss_1 = aes.losses["outputs"]
    loss_2 = aes.losses["outputs2"]

    tvs = model.trainable_variables
    accum_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
    train_loss_results = []

    for e in range(epochs):
        total_loss_1_avg = tf.keras.metrics.Mean()
        total_loss_2_avg = tf.keras.metrics.Mean()
        total_loss_avg = tf.keras.metrics.Mean()
        sim_doc_pos_avg = tf.keras.metrics.Mean()
        sim_doc_candi_avg = tf.keras.metrics.Mean()
        sim_doc_candj_avg = tf.keras.metrics.Mean()

        tqbar = tqdm(enumerate(train_generator), total=batches_per_epoch)

        for (n_batch, (inputs, outputs)) in tqbar:
            # En outputs["outputs2"] están los diff ranking para loss 2 #
            diff_rankings = outputs["outputs2"]
            logits, loss_value_1, loss_value_2, grads = match_get_grad(model, inputs, loss_1, loss_2, diff_rankings, training=True)

            for i, grad in enumerate(grads):
                if grad is not None:
                    accum_grads[i].assign_add(grad).__div__(grad_accum_iters)

            if n_batch % grad_accum_iters == 0:
                if n_batch > 0:
                    optimizer.apply_gradients(zip(accum_grads, model.trainable_variables))
                accum_grads = [var.assign(tf.zeros_like(var.initialized_value())) for var in accum_grads]


            sim_doc_pos = metric_funcs.cosine_similarity_pos_pair(None, logits[0])
            sim_doc_candi = metric_funcs.cosine_similarity_pos_pair(None, logits[1])
            sim_doc_candj = metric_funcs.cosine_similarity_neg_pair(None, logits[1])

            total_loss_1_avg.update_state(loss_value_1)
            total_loss_2_avg.update_state(loss_value_2)
            total_loss_avg.update_state(loss_value_1 + loss_value_2)

            sim_doc_pos_avg.update_state(sim_doc_pos)
            sim_doc_candi_avg.update_state(sim_doc_candi)
            sim_doc_candj_avg.update_state(sim_doc_candj)

            # Monitorize losses and metrics after the batch #
            tqbar.set_description("Total loss: %.4f: Loss1: %.4f Loss2: %.4f |||" \
                  "Doc-Pos sim: %.4f, Doc-Candi sim: %.4f, Doc-Candj sim: %.4f\n"%
                  (total_loss_avg.result(),
                   total_loss_1_avg.result(),
                   total_loss_2_avg.result(), sim_doc_pos_avg.result(),
                   sim_doc_candi_avg.result(), sim_doc_candj_avg.result()))
            tqbar.refresh()

            if hour_checkpointing:
                save_flag = utils.diff_hours(prev_time)
                if save_flag:
                    prev_time = datetime.utcnow()
                    aes.save_model(model, path_save_weights)

        train_loss_results.append(total_loss_avg.result())

        # Monitorize losses and metrics after the epoch #
        tqbar.set_description("Total loss: %.4f: Loss1: %.4f Loss2: %.4f |||" \
                              "Doc-Pos sim: %.4f, Doc-Candi sim: %.4f, Doc-Candj sim: %.4f\n" %
                              (total_loss_avg.result(),
                               total_loss_1_avg.result(),
                               total_loss_2_avg.result(), sim_doc_pos_avg.result(),
                               sim_doc_candi_avg.result(), sim_doc_candj_avg.result()))
        tqbar.refresh()

