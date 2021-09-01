import dataset
import pandas as pd
from tqdm import tqdm
import glob
import os
from pyntcloud import PyntCloud


def eval(config):
  """Evaluate point clouds compressed with residual encoder."""

  # Gets all original, normal and base layer decompressed files
  ori_files = sorted([f for f in os.listdir(config["ori_dir"]) if '.ply' in f])
  nor_files = sorted([f for f in os.listdir(config["nor_dir"]) if '.ply' in f])
  dist_files_base_layer = sorted(glob.glob(config["dist_pc_models_glob"]))

  # Reads csv file with bpp of point clouds compressed with base layer
  bin_bytes_base_layer_table = pd.read_csv(config["bin_csv_file"], index_col=0)

  # Gets all residual binaries and decompressed files 
  bin_files = sorted([f for f in os.listdir(config["merged_bin_dir"]) if '.bin' in f])
  dec_files = sorted([f for f in os.listdir(config["merged_dec_dir"]) if '.ply' in f])

  assert len(ori_files) == len(nor_files), "Amount of original PLY files is not the same as amount of PLY files with normals"
  assert len(ori_files) == len(bin_files), "Amount of original PLY files is not the same as amount of compressed residual BIN files"
  assert len(ori_files) == len(dec_files), "Amount of original PLY files is not the same as amount of decompressed residual PLY files"
  assert len(ori_files) == len(dist_files_base_layer), "Amount of original PLY files is not the same as amount of decompressed base layer PLY files"

  # Creates dataframe for saving metrics later as csv
  columns_list = ["name", "points_original", "points_decompressed", "points_base_layer", "bin_bytes_residual", "bpp_residual", "bin_bytes_base_layer",\
   "bpp_base_layer", "bin_bytes_total", "bpp_total", "g_metric_D1_base_layer", "g_metric_D2_base_layer", "g_metric_D1_residual", "g_metric_D2_residual"]

  bitrate_df = pd.DataFrame(columns=columns_list)

  print("Evaluating point clouds from the test set...")
  for ori_file, nor_file, bin_file, dec_file, dist_path_base_layer in tqdm(zip(ori_files, nor_files, bin_files, dec_files, dist_files_base_layer)):

    # Get full path of files
    ori_pc_path = os.path.join(config["ori_dir"], ori_file)
    nor_pc_path = os.path.join(config["nor_dir"], nor_file)
    dec_pc_path = os.path.join(config["merged_dec_dir"], dec_file)
    bin_path = os.path.join(config["merged_bin_dir"], bin_file)
    
    # Gets number of points in original, decompressed residual and decompressed base layer point cloud
    ori_pc = PyntCloud.from_file(ori_pc_path)
    ori_num_points = ori_pc.points.shape[0]
    dec_pc = PyntCloud.from_file(dec_pc_path)
    dec_num_points = dec_pc.points.shape[0]
    dist_pc_base_layer = PyntCloud.from_file(dist_path_base_layer)
    dist_base_layer_points = dist_pc_base_layer.points.shape[0]

    # Calculates bit-rate of residual representation
    bin_bytes = os.stat(bin_path).st_size
    bpp = bin_bytes * 8. / ori_num_points

    # Reads bit-rate of base layer compressed representation
    row_bin_table_dist_pc = bin_bytes_base_layer_table.loc[bin_bytes_base_layer_table["filename"] == os.path.split(dist_path_base_layer)[1]]
    bin_bytes_base_layer = row_bin_table_dist_pc["geo_bytes"].iloc[0]
    bpp_base_layer = bin_bytes_base_layer * 8. / ori_num_points

    #Gets resolution of point cloud
    if 'vox9' in ori_file:
        resolution = 511
    else:
        resolution = 1023

    #Calculates metrics between residual decompressed and original point cloud
    g_metric_D1, g_metric_D2 = evaluate_pc(config["pc_error"], ori_pc_path, dec_pc_path, nor_pc_path, resolution)

    #Calculates metrics between base layer decompressed and original point cloud
    g_metric_D1_base_layer, g_metric_D2_base_layer = evaluate_pc(config["pc_error"], ori_pc_path, dist_path_base_layer, nor_pc_path, resolution)
    
    ori_num_points = int(ori_num_points)
    dec_num_points = int(dec_num_points)
    bin_bytes = int(bin_bytes)
    dist_base_layer_points = int(dist_base_layer_points)

    data_list = [[ori_file, ori_num_points, dec_num_points, dist_base_layer_points, bin_bytes, bpp, bin_bytes_base_layer, bpp_base_layer, \
    bin_bytes + bin_bytes_base_layer, bpp + bpp_base_layer, g_metric_D1_base_layer, g_metric_D2_base_layer, g_metric_D1, g_metric_D2]]
    
    bitrate_df = bitrate_df.append(pd.DataFrame(columns=columns_list, data=data_list))

  #Saves everything in a csv file
  bitrate_df.to_csv(config["eval_csv_file"])

def evaluate_pc(pc_error_path, ori_path, dec_path, nor_path, resolution):
    """Evaluates a distorted point cloud model with metrics D1 and D2 PSNR.""" 
       
    p2po_psnr = p2pl_psnr = None
    os.system(f'./{pc_error_path} -a {ori_path} -b {dec_path} -n {nor_path} --color=1 --resolution={resolution} > tmp.log')
    with open('tmp.log', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ('mseF,PSNR' in line) and ('(p2point):' in line):
                p2po_psnr = float(line.split(':')[-1])
            elif ('mseF,PSNR' in line) and ('(p2plane):' in line):
                p2pl_psnr = float(line.split(':')[-1])
        
        if p2po_psnr is not None:
            g_metric_D1 = p2po_psnr
        else:
            g_metric_D1 = -1
        
        if p2pl_psnr is not None:
            g_metric_D2 = p2pl_psnr
        else:
            g_metric_D2 = -1

    return g_metric_D1, g_metric_D2
