"""
This file contains code adpated from https://github.com/mauriceqch/pcc_geo_cnn and is licensed under the MIT License
"""

import tensorflow as tf 
import glob
import pc_io
import random
import os
import zipfile
import numpy as np

def pc_to_tf(points, dense_tensor_shape):
    """Converts point cloud block into sparse tensor."""

    x = points

    paddings = [[0, 0], [0, 1]]

    geo_indices = tf.pad(x[:, :3], paddings, constant_values=0)

    indices = tf.cast(geo_indices, tf.int64)
    values = tf.ones_like(x[:, 0])

    st = tf.sparse.SparseTensor(indices, values, dense_tensor_shape)

    return st


def process_x(x, dense_tensor_shape):
    """Converts sparse tensor into dense tensor."""

    x = tf.sparse.to_dense(x, default_value=0, validate_indices=False)
    x.set_shape(dense_tensor_shape)
    x = tf.cast(x, tf.float32)

    return x

def input_fn(features, batch_size, dense_tensor_shape, preprocess_threads, prefetch_size=1):
    """Create input data pipeline."""

    with tf.device('/cpu:0'):

        # Creates dataset
        gen = lambda : iter(features)
        dataset = tf.data.Dataset.from_generator(gen, tf.float32, tf.TensorShape([None, 3]))

        # Converts point cloud blocks into occupancy maps as dense tensors 
        dataset = dataset.map(lambda x: pc_to_tf(x, dense_tensor_shape), num_parallel_calls=preprocess_threads)
        dataset = dataset.map(lambda x: process_x(x, dense_tensor_shape), num_parallel_calls=preprocess_threads)

        # Separates dataset into batches
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_size)

    return dataset


def short_filename(block_dist_filepath):
    """Receives as input a filepath for a distorted block and outputs its equivalent reference filepath

    Distorted filename convention: {model}_{vox}_{codec}_{compression_level}_{pos}.ply
    Reference filename convention: {model}_{vox}_{pos}.ply"""

    filename = os.path.split(block_dist_filepath)[1]
    filename_parts = filename.split(sep="_")

    filename_parts = [filename_parts[0], filename_parts[1], filename_parts[4]]
    filename = "_".join(filename_parts)

    return filename

def get_dataset(config):
    """Loads training and validation dataset from disk."""

    train_set_ref_glob = config["train_set_ref_glob"]
    train_set_dist_glob = config["train_set_dist_glob"]

    # Goes through the training set and read the point cloud blocks
    block_ref_file_list = glob.glob(train_set_ref_glob)
    block_dist_file_list = glob.glob(train_set_dist_glob)

    # Excludes reference blocks that don't have a pair in the list of distorted blocks
    block_dist_file_short_list = [short_filename(x) for x in block_dist_file_list]
    block_ref_file_list = [x for x in block_ref_file_list if os.path.split(x)[1] in block_dist_file_short_list]

    # Excludes distorted blocks that don't have a pair in the list of reference blocks
    block_ref_file_short_list = [os.path.split(x)[1] for x in block_ref_file_list]
    block_dist_file_list = [x for x in block_dist_file_list if short_filename(x) in block_ref_file_short_list]

    # Sorts both lists so that blocks will match
    block_ref_file_list = sorted(block_ref_file_list)
    block_dist_file_list = sorted(block_dist_file_list)

    total_dataset_size = len(block_ref_file_list)
    print(f"Total training set size: {total_dataset_size}")

    # Does a random permutation on both lists
    perm_vec = np.random.permutation(total_dataset_size)
    block_ref_file_list = np.array(block_ref_file_list)[perm_vec]
    block_dist_file_list = np.array(block_dist_file_list)[perm_vec]

    # Splits file list into train and validation
    train_ref_file_list = block_ref_file_list[:config["train_size"]]
    train_dist_file_list = block_dist_file_list[:config["train_size"]]
    val_ref_file_list = block_ref_file_list[-config["val_size"]:]
    val_dist_file_list = block_dist_file_list[-config["val_size"]:]

    # Loads point cloud blocks from disk
    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(config["block_size_train"])
    points_train_ref = pc_io.load_points(train_ref_file_list, p_min, p_max,)
    points_train_dist = pc_io.load_points(train_dist_file_list, p_min, p_max)
    points_val_ref = pc_io.load_points(val_ref_file_list, p_min, p_max)
    points_val_dist = pc_io.load_points(val_dist_file_list, p_min, p_max)

    # Creates dataset objects
    train_ref_dataset = input_fn(points_train_ref, config["batch_size_train"], dense_tensor_shape, config["preprocess_threads"])
    train_dist_dataset = input_fn(points_train_dist, config["batch_size_train"], dense_tensor_shape, config["preprocess_threads"])
    val_ref_dataset = input_fn(points_val_ref, config["batch_size_train"], dense_tensor_shape, config["preprocess_threads"])
    val_dist_dataset = input_fn(points_val_dist, config["batch_size_train"], dense_tensor_shape, config["preprocess_threads"])

    # Zips distorted and reference blocks together into one dataset
    train_dataset = tf.data.Dataset.zip((train_ref_dataset, train_dist_dataset))
    val_dataset = tf.data.Dataset.zip((val_ref_dataset, val_dist_dataset))

    # Concatenates distorted and reference blocks into one block with two channels
    train_dataset = train_dataset.map(lambda x, y: tf.concat([x, y], 4))
    val_dataset = val_dataset.map(lambda x, y: tf.concat([x, y], 4))

    # Sets training dataset to shuffle and repeat after each iteration
    train_dataset = train_dataset.shuffle(buffer_size=config["train_size"], reshuffle_each_iteration=True)
    train_dataset = train_dataset.repeat()

    # Sets validation dataset to shuffle after each iteration
    val_dataset = val_dataset.shuffle(buffer_size=config["val_size"])

    return train_dataset, val_dataset

def get_test_dataset(config):
    """Loads test dataset from disk."""

    test_set_ref_glob = config["ref_pc_glob"]
    test_set_dist_glob = config["dist_pc_glob"]

    # Goes through the training set and read the point cloud blocks
    ref_pc_list = glob.glob(test_set_ref_glob)
    dist_pc_list = glob.glob(test_set_dist_glob)

    # Excludes reference blocks that don't have a pair in the list of distorted blocks
    dist_pc_list_short = [short_filename(x) for x in dist_pc_list]
    ref_pc_list = [x for x in ref_pc_list if os.path.split(x)[1] in dist_pc_list_short]

    # Excludes distorted blocks that don't have a pair in the list of reference blocks
    ref_pc_list_short = [os.path.split(x)[1] for x in ref_pc_list]
    dist_pc_list = [x for x in dist_pc_list if short_filename(x) in ref_pc_list_short]

    # Sorts both lists so that blocks will match
    ref_pc_list = sorted(ref_pc_list)
    dist_pc_list = sorted(dist_pc_list)

    # Loads point cloud blocks from disk
    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(config["block_size_test"])
    points_test_ref = pc_io.load_points(ref_pc_list, p_min, p_max)
    points_test_dist = pc_io.load_points(dist_pc_list, p_min, p_max)

    # Creates dataset objects
    test_ref_dataset = input_fn(points_test_ref, config["batch_size_test"], dense_tensor_shape, config["preprocess_threads"])
    test_dist_dataset = input_fn(points_test_dist, config["batch_size_test"], dense_tensor_shape, config["preprocess_threads"])

    # Zips distorted and reference blocks together into one dataset
    test_dataset = tf.data.Dataset.zip((test_ref_dataset, test_dist_dataset))

    # Concatenates distorted and reference blocks into one block with two channels
    test_dataset = test_dataset.map(lambda x, y: tf.concat([x, y], 4))

    return test_dataset, dist_pc_list


