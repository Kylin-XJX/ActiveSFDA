"""
Data preprocessing: generate surface variation dataset
"""

import os
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from torch import threshold
from pyntcloud import PyntCloud
from tqdm import tqdm


def boundary_extractor_nuScenes():

    pkl_path = ''
    save_path = ''
    threshold = 0.1

    print('pkl_path:',pkl_path)
    print('save_path:',save_path)
    print('threshold:',threshold)
    # define hyperparameters
    k_n = 50
    clmns = ['x', 'y', 'z',]

    data = []
    edges = []

    print(f'Loading {pkl_path}')
    with open(pkl_path, 'rb') as f:
        data.extend(pickle.load(f))

    for data_item in tqdm(data):
        points = data_item['points']

        pcd_pd = pd.DataFrame(data=points, columns=clmns)
        pcd1 = PyntCloud(pcd_pd)

        # find neighbors
        kdtree_id = pcd1.add_structure("kdtree")
        k_neighbors = pcd1.get_neighbors(k=k_n, kdtree=kdtree_id)

        # calculate eigenvalues
        pcd1.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

        e1 = pcd1.points['e3('+str(k_n+1)+')'].values
        e2 = pcd1.points['e2('+str(k_n+1)+')'].values
        e3 = pcd1.points['e1('+str(k_n+1)+')'].values

        sum_eg = np.add(np.add(e1, e2), e3)
        sigma = np.divide(e1, sum_eg)
        sigma_value = sigma
        if threshold is not None:
            sigma_value[sigma_value > threshold] = threshold

        edges.append(sigma_value)
    save_dir = os.path.dirname(save_path)
    if not os.path.isdir(save_dir):
        warnings.warn('Make a new directory: {}'.format(save_dir))
        os.makedirs(save_dir,exist_ok=True)
    np.save(save_path,edges)
    print(f'save label mask to {save_path} !')


def boundary_extractor_semkitti():

    points_path = ''
    save_path = ''
    threshold = 0.1

    print('points_path:',points_path)
    print('save_path:',save_path)
    print('threshold:',threshold)
    # define hyperparameters
    k_n = 50
    clmns = ['x', 'y', 'z',]

    edges = []

    print("Loading points")
    points_data = []
    points_data.extend(np.load(points_path, allow_pickle=True))

    for points in tqdm(points_data):

        pcd_pd = pd.DataFrame(data=points, columns=clmns)
        pcd1 = PyntCloud(pcd_pd)

        # find neighbors
        kdtree_id = pcd1.add_structure("kdtree")
        k_neighbors = pcd1.get_neighbors(k=k_n, kdtree=kdtree_id)

        # calculate eigenvalues
        pcd1.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

        e1 = pcd1.points['e3('+str(k_n+1)+')'].values
        e2 = pcd1.points['e2('+str(k_n+1)+')'].values
        e3 = pcd1.points['e1('+str(k_n+1)+')'].values

        sum_eg = np.add(np.add(e1, e2), e3)
        sigma = np.divide(e1, sum_eg)
        sigma_value = sigma
        if threshold is not None:
            sigma_value[sigma_value > threshold] = threshold

        edges.append(sigma_value)
    save_dir = os.path.dirname(save_path)
    if not os.path.isdir(save_dir):
        warnings.warn('Make a new directory: {}'.format(save_dir))
        os.makedirs(save_dir,exist_ok=True)
    np.save(save_path,edges)
    print(f'save label mask to {save_path} !')

def main():

    boundary_extractor_semkitti()
    # boundary_extractor_nuScenes()
    # print(globals())

if __name__ == '__main__':
    main()