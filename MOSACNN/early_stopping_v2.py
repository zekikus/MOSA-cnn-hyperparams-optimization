import keras as k

class EarlyStoppingV2(k.callbacks.Callback):
  """
  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

  def __init__(self, patience=0):
      super(EarlyStoppingV2, self).__init__()
      self.patience = patience
      # best_weights to store the weights at which the minimum loss occurs.
      self.best_loss = None
      self.worst_counter = 0

  def on_train_begin(self, logs=None):
      # The number of epoch it has waited when loss is no longer minimum.
      self.epoch = 0
      # The epoch the training stops at.
      self.stopped_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
      valid_loss = logs.get("val_loss")

      if epoch == 0:
        self.best_loss = valid_loss

      if self.best_loss < valid_loss:
        self.worst_counter = self.worst_counter + 1
      else:
        self.worst_counter = 0
        self.best_loss = valid_loss

      if self.worst_counter >= 3:
        self.stopped_epoch = epoch
        self.model.stop_training = True
      else:
        print(f"Devam: {self.worst_counter} - {self.best_loss} - {epoch}")

  def on_train_end(self, logs=None):
      if self.stopped_epoch > 0:
          print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))