"""
This file contains code adpated from https://github.com/tensorflow/compression and is licensed under the Apache License 2.0
"""

import compression_model
import traceback
import tensorflow as tf
import dataset

def train(config):
  """Instantiates and trains the model."""

  model = compression_model.ResPCModel(config)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
  )

  train_dataset, validation_dataset = dataset.get_dataset(config)

  model.fit(
      train_dataset.prefetch(8),
      epochs=config["epochs"],
      steps_per_epoch=config["steps_per_epoch"],
      validation_data=validation_dataset.cache(),
      validation_freq=1,
      callbacks=[
          tf.keras.callbacks.TerminateOnNaN(),
          tf.keras.callbacks.TensorBoard(
              log_dir=config["checkpoint_dir"],
              histogram_freq=1, update_freq="epoch"),
          tf.keras.callbacks.experimental.BackupAndRestore(config["checkpoint_dir"]),
      ],
      verbose='auto',
  )

  #Saves model
  model.save(config["model_dir"])
  print("Model saved!")
