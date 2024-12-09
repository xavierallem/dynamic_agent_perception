# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from opencood.utils import pcd_utils
from spconv.utils import VoxelGeneratorV2

class BasePreprocessor(object):
    """
    Basic Lidar pre-processor with graph structure integration.

    Parameters
    ----------
    preprocess_params : dict
        The dictionary containing all parameters of the preprocessing.

    train : bool
        Train or test mode.
    """

    def __init__(self, preprocess_params, train):
        self.params = preprocess_params
        self.train = train

    def preprocess(self, pcd_np):
        """
        Preprocess the lidar points by creating a graph of pillars.

        Parameters
        ----------
        pcd_np : np.ndarray
            The raw lidar points.

        Returns
        -------
        data_dict : dict
            The output dictionary containing the graph data.
        """
        # Downsample lidar points
        sample_num = self.params['args']['sample_num']
        pcd_np = pcd_utils.downsample_lidar(pcd_np, sample_num)

        # Define voxel grid parameters
        voxel_size = self.params['args']['voxel_size']
        pc_range = self.params["cav_lidar_range"]
        max_points_per_voxel = self.params['args']['max_points_per_voxel']
        max_voxels = self.params['args']['max_voxel_train'] if self.train else self.params['args']['max_voxel_test']

        # Create voxel generator
        
        voxel_generator = VoxelGeneratorV2(voxel_size=voxel_size,
                                           point_cloud_range=pc_range,
                                           max_num_points=max_points_per_voxel,
                                           max_voxels=max_voxels)

        # Generate voxels
        voxel_output = voxel_generator.generate(pcd_np)
        voxels, coordinates, num_points = voxel_output

        # Convert to torch tensors
        voxels = torch.from_numpy(voxels)
        coordinates = torch.from_numpy(coordinates)
        num_points = torch.from_numpy(num_points)

        # Create node features (e.g., mean of points in each voxel)
        node_features = voxels.mean(dim=1)

        # Create edge indices using k-NN
        k = self.params['args']['k_nearest_neighbors']
        edge_index = knn_graph(node_features, k=k, batch=coordinates[:, 0])

        # Create graph
        graph = Data(x=node_features,
                     pos=coordinates[:, 1:],
                     edge_index=edge_index,
                     batch=coordinates[:, 0])

        # Prepare data dictionary
        data_dict = {
            'graph': graph,
            'downsample_lidar': pcd_np
        }

        return data_dict

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points with shape
            (img_row, img_col).
        """
        L1, W1, H1, L2, W2, H2 = self.params["cav_lidar_range"]
        img_row = int((L2 - L1) / ratio)
        img_col = int((W2 - W1) / ratio)
        bev_map = np.zeros((img_row, img_col))
        bev_origin = np.array([L1, W1, H1]).reshape(1, -1)
        # (N, 3)
        indices = ((points[:, :3] - bev_origin) / ratio).astype(int)
        mask = np.logical_and(indices[:, 0] > 0, indices[:, 0] < img_row)
        mask = np.logical_and(mask, np.logical_and(indices[:, 1] > 0,
                                                   indices[:, 1] < img_col))
        indices = indices[mask, :]
        bev_map[indices[:, 0], indices[:, 1]] = 1
        return bev_map