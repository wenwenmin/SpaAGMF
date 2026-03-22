
import os
import random
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
from scipy.spatial import KDTree
from torch.utils.data import Dataset

from utils import random_patch_sampling, random_neighbor_sampling


class MineDataset(Dataset):
    def __init__(self, data, mode, cfg, is_train=True):
        """
        initialize

        Parameters:
            names:
            cfg:
        """
        self.data = data
        self.mode = mode
        self.is_train = is_train
        self.cfg = cfg

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.data['patch'].shape[0]

    def __getitem__(self, idx):
        """
        Returns the pre-processed data for the specified index.

        Parameters:
            idx (int): Index of the sample to return.

        Returns:
            dict: {
                patch:
                gene:
                cancer:
            }
        """
        if self.mode == 'ctt':
            return {
                'cls': self.data['patch'][idx, 0],
                'gene': self.data['gene'][idx],
                'label': self.data['cancer'][idx],
            }
        else :
            indices = self.data['indices'][idx]
            if self.is_train:
                indices = random_neighbor_sampling(
                    indices,
                    self.cfg.augment.fixed_num,
                    self.cfg.augment.random_num
                )
            # else:
            #     indices = indices[:self.cfg.augment.fixed_num]

            patch_tokens = self.data['patch'][idx]
            if self.is_train:
                patch_tokens = random_patch_sampling(
                    patch_tokens,
                    keep_num=self.cfg.augment.patch_keep
                )

            return {
                'cls': self.data['patch'][indices][:, 0, :],
                'patch_tokens': patch_tokens,
                'gene': self.data['gene'][indices],
                'label': self.data['cancer'][idx],
                'spatial': self.data['spatial'][indices],
                'pos_weight':self.data['pos_weight']
            }

def load_data(names, cfg):
    """
    Load the pre-processed data from `.h5ad` files.
    """
    patch_list, exp_list, cancer_list, index_list, spatial_list = [], [], [], [], []
    offset = 0
    for name in names:
        save_path = Path(cfg.dataset.data_dir) / f"{name}.h5ad"
        adata = sc.read_h5ad(save_path)
        num = adata.shape[0]
        patch_list.extend(adata.obsm['patch'])
        exp_list.extend(adata.X)
        cancer_list.extend(adata.obs['cancer'])
        indices = get_neighbors(adata, cfg.dataset.k_neighbors) + offset
        index_list.extend(indices)
        spatial_list.extend(adata.obsm['spatial'])

        offset += num


    cancer_np = np.array(cancer_list)
    # 计算 pos_weight
    num_pos = (cancer_np == 1).sum()
    num_neg = (cancer_np == 0).sum()
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0

    return {
        'patch': torch.from_numpy(np.array(patch_list)),
        'gene': torch.from_numpy(np.array(exp_list)).float(),
        'cancer': torch.from_numpy(np.array(cancer_list)).long().reshape(-1, 1),
        'indices': torch.from_numpy(np.array(index_list)).long(),
        'spatial': torch.from_numpy(np.array(spatial_list)).float(),  # 新增坐标
        'pos_weight': torch.tensor(pos_weight, dtype=torch.float32),
    }

def get_neighbors(adata, k=3):
    """
    Get the k nearest neighbors based on spatial proximity or another criterion.

    Parameters:
        sample_name (str): Name of the sample.
        k (int): Number of nearest neighbors to fetch.

    Returns:
        list: A list of sample names representing the k nearest neighbors.
    """
    # Need spatial is a numpy array
    spatial = adata.obsm['spatial']

    tree = KDTree(spatial)
    distances, indices = tree.query(spatial, k=k + 1)

    # Remove the point itself (distance 0, index 0)
    return indices