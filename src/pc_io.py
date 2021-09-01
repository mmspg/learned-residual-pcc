"""
This file contains code adpated from https://github.com/mauriceqch/pcc_geo_cnn and is licensed under the MIT License
"""

import functools
from glob import glob
import logging
import multiprocessing

import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from tqdm import tqdm


logger = logging.getLogger(__name__)


class PC:
    def __init__(self, points, p_min, p_max):
        self.points = points
        self.p_max = p_max
        self.p_min = p_min
        self.data = {}

        assert np.all(p_min < p_max), f"p_min <= p_max must be true : p_min {p_min}, p_max {p_max}"
        assert np.all(points[:, :3] < p_max), f"points must be inferior to p_max {p_max}"
        assert np.all(points[:, :3] >= p_min), f"points must be superior to p_min {p_min}"

    def __repr__(self):
        return f"<PC with {self.points.shape[0]} points (p_min: {self.p_min}, p_max: {self.p_max})>"

    def is_empty(self):
        return self.points.shape[0] == 0

    def p_mid(self):
        p_min = self.p_min
        p_max = self.p_max
        return p_min + ((p_max - p_min) / 2.)


def df_to_pc(df, p_min, p_max):
    points = df[['x', 'y', 'z']].values

    return PC(points, p_min, p_max)


def pa_to_df(points):
    if len(points) == 0:
        df = pd.DataFrame(
            data={
                'x': [],
                'y': [],
                'z': []})
    else:
        df = pd.DataFrame(
            data={
                'x': points[:, 0],
                'y': points[:, 1],
                'z': points[:, 2]}, dtype=np.float32)

    return df


def pc_to_df(pc):
    points = pc.points
    return pa_to_df(points)


def load_pc_obj(path, p_min, p_max):
    logger.debug(f"Loading PC {path}")
    pc = PyntCloud.from_file(path)
    ret = df_to_pc(pc.points, p_min, p_max)
    logger.debug(f"Loaded PC {path}")
    return ret

def load_pc(path):
    pc = PyntCloud.from_file(path)
    pc_df = pc.points
    pc_np = pc_df[['x', 'y', 'z']].values

    return pc_np


def get_shape_data(resolution):
    bbox_min = 0
    bbox_max = resolution
    p_max = np.array([bbox_max, bbox_max, bbox_max])
    p_min = np.array([bbox_min, bbox_min, bbox_min])
    dense_tensor_shape = np.concatenate([[1], p_max]).astype('int64')

    dense_tensor_shape = dense_tensor_shape[[1, 2, 3, 0]]

    return p_min, p_max, dense_tensor_shape

def load_points_func(x, p_min, p_max):
    return load_pc_obj(x, p_min, p_max).points

def export_occ_map(occ_map, filename):
    """Takes as input occupancy maps with no added dimensions and saves it in disk."""
    pc_np = np.argwhere(occ_map)
    points = pd.DataFrame(data=pc_np, columns=["x", "y", "z"], dtype=np.float32)
    pc = PyntCloud(points)
    pc.to_file(filename)
    


def load_points(files, p_min, p_max, batch_size=32):
    files_len = len(files)

    with multiprocessing.Pool() as p:
        logger.info('Loading PCs into memory (parallel reading)')
        f = functools.partial(load_points_func, p_min=p_min, p_max=p_max)
        points = np.array(list(tqdm(p.imap(f, files, batch_size), total=files_len)))

    return points

def load_occ_map(filename, resolution):
    """Loads block from original point cloud."""

    pc_np = load_pc(filename)
    pc_np_geo = pc_np[:, :3]
    pc_np_ind = pc_np_geo.astype(int)

    pc_np_vox = np.zeros((resolution, resolution, resolution, 1))
    pc_np_vox[pc_np_ind[:,0], pc_np_ind[:,1], pc_np_ind[:,2], 0] = 1

    return pc_np_vox

