#!/usr/bin/env python

import train
import compress
import merge
import eval_pc
import argparse
import shutil
import os
import yaml

def set_auto_paths(config, config_path):
  """Defines paths for saving all files associated with trained model."""
  
  # Creates root directory for all compression models
  os.makedirs(config["output_root"], exist_ok=True)

  # Defines name for root directory of compression model
  if config["use_cfg_filename_as_modelname"]:
    model_name = os.path.split(config_path)[1].replace(".yml","").replace("_cfg","")
  else:
    all_models = os.listdir(config["output_root"])
    i = 1
    model_name =  "residual_model_" + str(i)
    while model_name in all_models:
      i = i + 1
      model_name =  "residual_model_" + str(i)

  # Creates root directory for compression model
  model_root = os.path.join(config["output_root"], model_name)
  os.makedirs(model_root, exist_ok=True)

  #Defines directories for checkpoint, saved model and output folders for compressed and decompressed test sets
  config["checkpoint_dir"] = os.path.join(model_root, config["checkpoint_rel_dir"])
  config["model_dir"] = os.path.join(model_root, config["model_rel_dir"])
  config["div_bin_dir"] = os.path.join(model_root, config["div_bin_rel_dir"])
  config["div_dec_dir"] = os.path.join(model_root, config["div_dec_rel_dir"])
  config["merged_bin_dir"] = os.path.join(model_root, config["merged_bin_rel_dir"])
  config["merged_dec_dir"] = os.path.join(model_root, config["merged_dec_rel_dir"])
  config["eval_csv_file"] = os.path.join(model_root, config["eval_csv_file"])

  #Copies config file to root directory of compression model
  shutil.copyfile(config_path, os.path.join(model_root, "model.yml"))


def main(config, config_path):
  """Execute function depending on command defined on configuration file."""

  if config["command"] == "multi_config":
    #If command is multi_config, executes the configurations of all yml files listed. 
    for configFile in config["yml_files"]:
      #Try to open all files in list, keep going if an error is found while executing one file.
      try:
        with open(configFile) as f:
          config_in = yaml.safe_load(f)
        main(config_in, configFile)
      except Exception:
        traceback.print_exc()

  if config["command"] == "train":
    train.train(config)

  elif config["command"] == "compress_one":
    compress.compress_one(config)

  elif config["command"] == "decompress_one":
    compress.decompress_one(config)

  elif config["command"] == "compress_decompress_one":
    compress.compress_one(config)
    compress.decompress_one(config)

  elif config["command"] == "compress_dataset": 
    compress.compress_dataset_batch(config)

  elif config["command"] == "merge_bin":
    merge.merge_bin(config)

  elif config["command"] == "merge_dec":
    merge.merge_dec(config)

  elif config["command"] == "eval":
    eval_pc.eval(config)

  elif config["command"] == "train_and_evaluate":
    if config["auto_paths"]:
      set_auto_paths(config, config_path)
    train.train(config)
    compress.compress_dataset_batch(config)
    merge.merge_bin(config)
    merge.merge_dec(config)
    eval_pc.eval(config)

  elif config["command"] == "compress_and_evaluate":
    if config["auto_paths"]:
      set_auto_paths(config, config_path)
    compress.compress_dataset_batch(config)
    merge.merge_bin(config)
    merge.merge_dec(config)
    eval_pc.eval(config)


################################################################################
# Script
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Model for residual coding of point clouds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'config_file',
        help='Configuration file with YAML format')

    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Configuration file not found."

    with open(args.config_file) as f:
      config = yaml.safe_load(f)

    main(config, args.config_file)