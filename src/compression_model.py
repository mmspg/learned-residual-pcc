"""
This file contains code adpated from https://github.com/tensorflow/compression and is licensed under the Apache License 2.0
"""

import tensorflow as tf
import tensorflow_compression as tfc
from focal_loss import focal_loss

class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self, num_filters):
    super().__init__(name="analysis")
    self.add(tf.keras.layers.Conv3D(
        num_filters, (9, 9, 9), name="layer_0", strides=(2, 2, 2),
        padding="same", use_bias=True,
        activation=tf.nn.relu))
    self.add(tf.keras.layers.Conv3D(
        num_filters, (5, 5, 5), name="layer_1", strides=(2, 2, 2),
        padding="same", use_bias=True,
        activation=tf.nn.relu))
    self.add(tf.keras.layers.Conv3D(
        num_filters, (5, 5, 5), name="layer_2", strides=(2, 2, 2),
        padding="same", use_bias=True,
        activation=tf.nn.relu))


class SynthesisTransform(tf.keras.Sequential):
  """The synthesis transform."""

  def __init__(self, num_filters, num_channels_residual):
    super().__init__(name="synthesis")
    self.add(tf.keras.layers.Conv3DTranspose(
        num_filters, (5, 5, 5), name="layer_0", strides=(2, 2, 2),
        padding="same", use_bias=True,
        activation=tf.nn.relu))
    self.add(tf.keras.layers.Conv3DTranspose(
        num_filters, (5, 5, 5), name="layer_1", strides=(2, 2, 2),
        padding="same", use_bias=True,
        activation=tf.nn.relu))
    self.add(tf.keras.layers.Conv3DTranspose(
        num_channels_residual, (9, 9, 9), name="layer_2", strides=(2, 2, 2),
        padding="same", use_bias=True,
        activation=tf.nn.sigmoid))



class RefinerTransform(tf.keras.Sequential):
  """The refiner transform."""

  def __init__(self, num_filters, num_layers, kernel_size):
    super().__init__(name="refiner")

    for counter in range(num_layers - 1):
      self.add(tf.keras.layers.Conv3D(
          num_filters, (kernel_size[counter], kernel_size[counter], kernel_size[counter]), name="layer_" + str(counter),
          padding="same", use_bias=True,
          activation=tf.nn.relu))
    self.add(tf.keras.layers.Conv3D(
        1, (kernel_size[-1], kernel_size[-1], kernel_size[-1]), name="layer_" + str(num_layers - 1), 
        padding="same", use_bias=True,
        activation=tf.nn.sigmoid))

class HyperAnalysisTransform(tf.keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_analysis")
    self.add(tf.keras.layers.Conv3D(
        num_filters, (3, 3, 3), name="layer_0", strides=1,
        padding="same", use_bias=True,
        activation=tf.nn.relu))
    self.add(tf.keras.layers.Conv3D(
        num_filters, (3, 3, 3), name="layer_1", strides=2,
        padding="same", use_bias=True,
        activation=tf.nn.relu))
    self.add(tf.keras.layers.Conv3D(
        num_filters, (1, 1, 1), name="layer_2", strides=1,
        padding="same", use_bias=False,
        activation=None))


class HyperSynthesisTransform(tf.keras.Sequential):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_synthesis")
    self.add(tf.keras.layers.Conv3DTranspose(
        num_filters, (1, 1, 1), name="layer_0", strides=1,
        padding="same", use_bias=True, 
        activation=tf.nn.relu))
    self.add(tf.keras.layers.Conv3DTranspose(
        num_filters, (3, 3, 3), name="layer_1", strides=2,
        padding="same", use_bias=True,
        activation=tf.nn.relu))
    self.add(tf.keras.layers.Conv3DTranspose(
        num_filters, (3, 3, 3), name="layer_2", strides=1,
        padding="same", use_bias=True,
        activation=None))


class ResPCModel(tf.keras.Model):
  """Residual model class."""

  def __init__(self, config):

    super().__init__()
    
    self.lmbda = config["lambda"]
    self.gamma = config["gamma"]
    self.alpha = config["alpha"]

    self.num_scales = config["num_scales"]
    offset = tf.math.log(config["scale_min"])
    factor = (tf.math.log(config["scale_max"]) - tf.math.log(config["scale_min"])) / (
        config["num_scales"] - 1.)
    self.scale_fn = lambda i: tf.math.exp(offset + factor * i)

    self.analysis_transform = AnalysisTransform(config["num_filters"])
    self.synthesis_transform = SynthesisTransform(config["num_filters"], config["num_channels_residual"])
    self.refiner_transform = RefinerTransform(config["num_filters_refiner"], config["num_layers_refiner"],  config["kernel_size_refiner"])
    self.hyper_analysis_transform = HyperAnalysisTransform(config["num_filters"])
    self.hyper_synthesis_transform = HyperSynthesisTransform(config["num_filters"])
    
    self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(config["num_filters"],))

    self.build((None, None, None, None, 2))

  def call(self, x, training):
    """Computes rate and distortion losses."""
    entropy_model = tfc.LocationScaleIndexedEntropyModel(
       tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=4,
       compression=False)
    side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=4, compression=False)

    # Calculates compressed representation and hyperprior
    y = self.analysis_transform(x)
    z = self.hyper_analysis_transform(abs(y))
    z_hat, side_bits = side_entropy_model(z, training=training)
    indexes = self.hyper_synthesis_transform(z_hat)
    y_hat, bits = entropy_model(y, indexes, training=training)

    # Separates reference from distorted occupancy map
    x_ref = x[:,:,:,:, :1]
    x_dist = x[:,:,:,:, 1:]

    # Computes residual representation and decoded probability map
    r = self.synthesis_transform(y_hat)
    r_cat = tf.keras.layers.concatenate([x_dist, r], axis=4)
    x_hat = self.refiner_transform(r_cat)

    # Total number of bits divided by total number of occupied voxels.
    num_voxels = tf.reduce_sum(x_ref[:, :, :, :, 0])
    bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_voxels
    
    # Computes focal loss
    distortion_geo = focal_loss(x_ref[:, :, :, :, 0], x_hat[:, :, :, :, 0], self.gamma, self.alpha) / num_voxels

    # Computes final loss value
    loss = bpp + self.lmbda * distortion_geo

    return loss, bpp, distortion_geo

  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, distortion_geo = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.distortion_geo.update_state(distortion_geo)

    return {m.name: m.result() for m in [self.loss, self.bpp, self.distortion_geo]}

  def test_step(self, x):
    loss, bpp, distortion_geo = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.distortion_geo.update_state(distortion_geo)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.distortion_geo]}

  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.distortion_geo = tf.keras.metrics.Mean(name="distortion_geo")

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    #After training, fix range coding tables.
    self.entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=4,
        compression=True)
    self.side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=4, compression=True)
    return retval

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
  ])
  def compress(self, x):
    """Compresses one point cloud block."""

    # Add batch dimension
    x = tf.expand_dims(x, 0)

    #Computes compressed representations
    y = self.analysis_transform(x)
    z = self.hyper_analysis_transform(abs(y))

    # Preserve spatial shapes of image and latents.
    x_shape = tf.shape(x)[1:-1]
    y_shape = tf.shape(y)[1:-1]
    z_shape = tf.shape(z)[1:-1]
    z_hat, _ = self.side_entropy_model(z, training=False)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :y_shape[2], :]
    side_string = self.side_entropy_model.compress(z)
    string = self.entropy_model.compress(y, indexes)
    return string, side_string, x_shape, y_shape, z_shape

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, None, None), dtype=tf.float32),
  ])
  def compress_batch(self, x):
    """Compresses the test set in batches."""

    #Computes compressed representations
    y = self.analysis_transform(x)
    z = self.hyper_analysis_transform(abs(y))

    # Preserve spatial shapes of image and latents
    x_shape = tf.shape(x)[1:-1]
    y_shape = tf.shape(y)[1:-1]
    z_shape = tf.shape(z)[1:-1]
    z_hat, _ = self.side_entropy_model(z, training=False)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :y_shape[2], :]
    side_string = self.side_entropy_model.compress(z)
    string = self.entropy_model.compress(y, indexes)
    return string, side_string, x_shape, y_shape, z_shape

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(3,), dtype=tf.int32),
      tf.TensorSpec(shape=(3,), dtype=tf.int32),
      tf.TensorSpec(shape=(3,), dtype=tf.int32),
  ])
  def decompress(self, x_dist, string, side_string, x_shape, y_shape, z_shape):
    """Decompresses one point cloud block."""

    z_hat = self.side_entropy_model.decompress(side_string, z_shape)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :y_shape[2], :]
    y_hat = self.entropy_model.decompress(string, indexes)
    
    #Add batch dimension
    x_dist = tf.expand_dims(x_dist, 0)

    #Computes residual representation
    r = self.synthesis_transform(y_hat)
    
    # Crop away any extraneous padding.
    r = r[:, :x_shape[0], :x_shape[1], :x_shape[2], :]

    # Concatenate distorted model with residual representation
    r_cat = tf.concat([x_dist, r], 4)

    # Refine distorted model
    x_hat = self.refiner_transform(r_cat)

    # Remove batch dimension
    x_hat = x_hat[0, :, :, :, :]

    # Round computed voxel occupancy probabilities to 0 or 1 using 0.5 as threshold
    x_hat = tf.round(x_hat)
   
    return x_hat

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, None, None), dtype=tf.float32),
      tf.TensorSpec(shape=(None,), dtype=tf.string),
      tf.TensorSpec(shape=(None,), dtype=tf.string),
      tf.TensorSpec(shape=(3,), dtype=tf.int32),
      tf.TensorSpec(shape=(3,), dtype=tf.int32),
      tf.TensorSpec(shape=(3,), dtype=tf.int32),
  ])
  def decompress_batch(self, x_dist, string, side_string, x_shape, y_shape, z_shape):
    """Decompresses the test set in batches."""

    z_hat = self.side_entropy_model.decompress(side_string, z_shape)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :y_shape[2], :]
    y_hat = self.entropy_model.decompress(string, indexes)

    #Computes residual representation
    r = self.synthesis_transform(y_hat)
    
    # Crop away any extraneous padding.
    r = r[:, :x_shape[0], :x_shape[1], :x_shape[2], :]

    #Concatenate distorted model with residual representation
    r_cat = tf.concat([x_dist, r], 4)

    #Refine distorted model
    x_hat = self.refiner_transform(r_cat)

    # Round computed voxel occupancy probabilities to 0 or 1 using 0.5 as threshold
    x_hat = tf.round(x_hat)

    return x_hat
