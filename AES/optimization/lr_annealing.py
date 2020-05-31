from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.math import minimum

class Noam(LearningRateSchedule):
  def __init__(self, warmup_steps, hidden_dims, name=None):
      self.warmup_steps = warmup_steps
      self.hidden_dims = hidden_dims
      self.name = name
      super(Noam, self).__init__()

  def __call__(self, step):
      new_lr = 2e-3 * \
               minimum((step ** (-0.5)), step * (self.warmup_steps ** (-1.5)))
      return new_lr

  def get_config(self):
    return {
        "warmup_steps": self.warmup_steps,
        "hidden_dims": self.hidden_dims,
        "name": self.name
    }
