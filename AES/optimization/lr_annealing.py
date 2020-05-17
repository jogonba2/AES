from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class Noam(LearningRateSchedule):
  def __init__(self, warmup_steps, hidden_dims, name=None):
      self.warmup_steps = warmup_steps
      self.hidden_dims = hidden_dims
      super(Noam, self).__init__()

  def __call__(self, step):
      return (self.hidden_dims ** -0.5) * \
              min((step ** (-0.5)), step * (self.warmup_steps ** (-1.5)))

  def get_config(self):
    return {
        "warmup_steps": self.warmup_steps,
        "hidden_dims": self.hidden_dims,
        "name": self.name
    }