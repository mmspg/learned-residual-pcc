import dataset
from pyntcloud import PyntCloud
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

def merge_dec(config):
  """Merge decompressed blocks into whole point clouds for the entire test set."""
  
  ori_files = sorted(glob.glob(config["dist_pc_models_glob"]))
  div_files = sorted([f for f in os.listdir(config["div_dec_dir"]) if '.ply' in f])

  os.makedirs(config["merged_dec_dir"], exist_ok=True)

  #Merges decompressed ply files of point cloud blocks for each point cloud model from the test set
  print("Merging ply files of point cloud blocks from the test set...")
  for ori_path in tqdm(ori_files):
    ori_file = os.path.split(ori_path)[1]
    merge_pc_dec(ori_file, div_files, config["block_size_test"], config["div_dec_dir"], config["merged_dec_dir"])

def merge_pc_dec(ori_file, div_files, block_size, div_dir, output_dir):
  """Merge decompressed blocks from 'div_files' corresponding to the model from 'ori_file' list into one point cloud."""

  #Gets all ply files for that point cloud model
  cur_div_files = [f for f in div_files if ori_file[:-4] in f]

  #Initializes dataframe for point cloud
  points = pd.DataFrame(data={ 'x': [], 'y': [], 'z': [] })

  #Reads points cloud block and concatenates points to dataframe for model, adding offset defined by block indexes
  for div_file in cur_div_files:
    div_pc = PyntCloud.from_file(os.path.join(div_dir, div_file))
    div_pc_points = div_pc.points
    ind = block_indices(div_file)
    div_pc_points.x += ind[0] * block_size
    div_pc_points.y += ind[1] * block_size
    div_pc_points.z += ind[2] * block_size
    
    points = pd.concat([points, div_pc_points])
      
  points.reset_index(drop=True, inplace=True)

  res_pc = PyntCloud(points)
  res_pc.to_file(os.path.join(output_dir, f'{ori_file[:-4]}_residual.ply'))

def merge_bin(config):
  """Merge compressed blocks into whole binary files for the entire test set."""
  
  ori_files = sorted(glob.glob(config["dist_pc_models_glob"]))
  div_files = sorted([f for f in os.listdir(config["div_bin_dir"]) if '.bin' in f])

  os.makedirs(config["merged_bin_dir"], exist_ok=True)

  #Merges binary files of point cloud blocks for each point cloud model from the test set
  print("Merging binaries of point cloud blocks from the test set...")
  for ori_path in tqdm(ori_files):
    ori_file = os.path.split(ori_path)[1]
    merge_pc_bin(ori_file, div_files, config["block_size_test"], config["div_bin_dir"], config["merged_bin_dir"])


def merge_pc_bin(ori_file, div_files, block_size, div_dir, output_dir):
  """Merge compressed blocks from 'div_files' corresponding to the model from 'ori_file' list into one binary file."""
  
  # Gets all binary files for that point cloud model
  cur_div_files = [f for f in div_files if ori_file[:-4] in f]
  
  # Initializes bitstream
  bitstream = b''

  for div_file in cur_div_files:
    # Concatenates binary file with block index to main bitstream
    with open(os.path.join(div_dir,div_file), "rb") as f:
      index = block_indices(div_file)
      bits_block = f.read()
      bits_block = index.astype(np.uint8).tobytes() + bits_block
    bitstream = bitstream + bits_block

  with open(os.path.join(output_dir, ori_file[:-4] + "_residual.bin"), 'wb') as f:
    f.write(bitstream)

def block_indices(filename):
  """Gets block index from filename."""

  # Expected format: {point_cloud_model}_i{i_pos}_j{j_pos}_k{k_pos}_residual.{bin or ply}

  filename = filename.replace(".bin", "")
  filename = filename.replace(".ply", "")
  filename = filename.replace("_residual", "")

  indices = filename.split('i')[-1]
  i_val = int(indices.split('j')[0].replace('_',''))
  j_val = int((indices.split('j')[1]).split('k')[0].replace('_',''))  
  k_val = int((indices.split('k')[1]).split('.')[0].replace('_',''))

  return np.array([i_val, j_val, k_val])
