import numpy as np
import pc_io
import tensorflow as tf
import dataset
import os
from tqdm import tqdm
import multiprocessing
import functools

def compress_one(config):
  """Compresses one block with trained model."""

  # Load model used to compress the block. 
  model = tf.keras.models.load_model(config["model_dir"])

  # Load occupancy map from original point cloud block
  ref_pc_np_vox = pc_io.load_occ_map(config["ref_pc_file"], config["block_size_test"])
  # Load occupancy map from distorted point cloud block
  dist_pc_np_vox = pc_io.load_occ_map(config["dist_pc_file"], config["block_size_test"])

  # Concatenates reference and distorted block
  input_pc = np.concatenate((ref_pc_np_vox, dist_pc_np_vox), axis=3)

  #Compress input with loaded model
  tensors = model.compress(input_pc)

  # Write a binary file with the the string and the side string
  string = tensors[0].numpy()[0]
  side_string = tensors[1].numpy()[0]

  bitstream = pack_bitstream(string, side_string, config["bytes_length"])

  with open(config["bin_file"], "wb") as f:
    f.write(bitstream)

  print("Residual representation successfully compressed")

def compress_dataset_batch(config):
  """Compresses and decompresses all blocks from test set."""

  # Load model used to compress the block. 
  model = tf.keras.models.load_model(config["model_dir"])

  # Get test dataset
  test_dataset, dist_pc_list = dataset.get_test_dataset(config)

  # Creates folders for binary and decompressed files
  os.makedirs(config["div_bin_dir"], exist_ok=True)
  os.makedirs(config["div_dec_dir"], exist_ok=True)

  #Separates file list in batches
  gen_dist_pc = chunks(dist_pc_list, config["batch_size_test"])

  # Defines device used during compression 
  device = 'CPU' if config["force_cpu"] else 'GPU'

  num_channels = 1 

  print("Compressing and decompressing residual representations in the test set ...")
  
  for batch in tqdm(list(test_dataset)):

    # Compress and decompress batch with loaded model
    with tf.device(device):
      tensors = model.compress_batch(batch)
      x_hat_batch = model.decompress_batch(batch[:, :, :, :, num_channels:], *tensors)

    # Converts sequences of compressed blocks, compressed hyperpriors and decompressed blocks to lists
    string_list = list(tensors[0].numpy())
    side_string_list = list(tensors[1].numpy())
    x_hat_batch_list = list(x_hat_batch.numpy())

    iterable = zip(string_list, side_string_list, x_hat_batch_list, next(gen_dist_pc))

    # Writes files to disk using parallel computing
    with multiprocessing.Pool() as p:
      f = functools.partial(write_compressed_files, div_bin_dir=config["div_bin_dir"], div_dec_dir=config["div_dec_dir"], bytes_length=config["bytes_length"])
      p.map(f, iterable)

def decompress_one(config):
  """Decompresses one block with trained model."""

  # Load model used to decompress the block. 
  model = tf.keras.models.load_model(config["model_dir"])

  # Load occupancy map from distorted point cloud block
  dist_pc_np_vox = pc_io.load_occ_map(config["dist_pc_file"], config["block_size_test"])

  # Reads the binary file
  with open(config["bin_file"], "rb") as f:
    bitstream = f.read()

  # Unpacks the bitstream into compressed block and hyperprior
  string, side_string = unpack_bitstream(bitstream, config["bytes_length"])

  string_tensor = tf.convert_to_tensor([string])
  side_string_tensor = tf.convert_to_tensor([side_string])

  # Assemble strings and shapes to input to decompress method
  tensors = (string_tensor, side_string_tensor) + get_shapes(config["block_size_test"], model)

  # Decompress input with loaded model
  x_hat = model.decompress(dist_pc_np_vox, *tensors)

  #Exports decompressed block to disk
  pc_io.export_occ_map(np.squeeze(x_hat), config["dec_file"])

  print("Residual representation successfully decompressed")

def write_compressed_files(input_list, div_bin_dir, div_dec_dir, bytes_length):
  """Write compressed files to disk."""

  string, side_string, x_hat, dist_pc_path = input_list

  filename = os.path.split(dist_pc_path)[1]
  bin_file = os.path.join(div_bin_dir, filename.replace(".ply", "_residual.bin"))
  dec_file = os.path.join(div_dec_dir, filename.replace(".ply", "_residual.ply"))

  bitstream = pack_bitstream(string, side_string, bytes_length)

  with open(bin_file, "wb") as f:
    f.write(bitstream)

  pc_io.export_occ_map(np.squeeze(x_hat), dec_file)

def chunks(lst, n):
  """Generator that yields successive n-sized batches from list."""
  for i in range(0, len(lst), n):
      yield lst[i:i + n]

def pack_bitstream(string, side_string, bytes_length):
  """Pack compressed representation of block and hyperprior into one bitstream."""

  bitstream = len(string).to_bytes(bytes_length, byteorder='big') + string + \
    len(side_string).to_bytes(bytes_length, byteorder='big') + side_string

  return bitstream

def unpack_bitstream(bitstream, bytes_length):
  """Unpack bitstream into compressed representation of block and hyperprior."""

  string_length = int.from_bytes(bitstream[:bytes_length], "big")
  string = bitstream[bytes_length:][:string_length]

  side_string_length = int.from_bytes(bitstream[bytes_length+string_length:][:bytes_length], "big")
  side_string = bitstream[bytes_length+string_length+bytes_length:][:side_string_length]

  return string, side_string

def get_shapes(block_size, model):
  """Get shapes from uncompressed block, latent space block and hyperprior block."""
  
  #x_shape is the shape of the block
  x_shape = tf.convert_to_tensor(np.repeat(block_size, 3), dtype=tf.int32)

  #Calculates y_shape by dividing x_shape by the strides of each layer of the analysis transform
  strides_x = np.array([l.strides for l in model.analysis_transform.layers])
  x_reduction = np.prod(strides_x, axis=0)
  y_shape = tf.cast(x_shape/x_reduction, dtype=tf.int32)
  
  #Calculates z_shape by dividing y_shape by the strides of each layer of the hyper analysis transform
  strides_y =  np.array([l.strides for l in model.hyper_analysis_transform.layers])
  y_reduction = np.prod(strides_y, axis=0)
  
  z_shape = tf.cast(y_shape/y_reduction, dtype=tf.int32)

  return x_shape, y_shape, z_shape
