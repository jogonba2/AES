from tensorflow.keras.callbacks import Callback


class Noam(Callback):

    def __init__(self, warmup_steps, hidden_dims,
                 accum_iters, initial_batch):
        super().__init__()
        self.batch = initial_batch
        self.warmup_steps = warmup_steps
        self.hidden_dims = hidden_dims
        self.accum_iters = accum_iters

    def on_batch_end(self, _, logs={}):
        if (self.batch + 1) % self.accum_iters == 0:
            new_lr = (self.hidden_dims ** -0.5) * \
                      min((((self.batch+1) / self.accum_iters) ** (-0.5)),
                          ((self.batch+1) / self.accum_iters) *
                          (self.warmup_steps ** (-1.5)))
            self.model.optimizer.learning_rate.assign(new_lr)
        self.batch += 1
