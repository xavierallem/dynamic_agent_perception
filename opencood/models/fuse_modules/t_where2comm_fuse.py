"""
Implementation of Where2comm fusion.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention



class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        # Threshold of objectiveness
        self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, B):
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
        """

        _, _, H, W = batch_confidence_maps[0].shape

        communication_masks = []
        communication_rates = []
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            L = communication_maps.shape[0]
            if self.training:
                # Official training proxy objective
                K = int(H * W * random.uniform(0, 1))
                communication_maps = communication_maps.reshape(L, H * W)
                _, indices = torch.topk(communication_maps, k=K, sorted=False)
                communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
                communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
            elif self.threshold:
                ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
            else:
                communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)

            communication_rate = communication_mask.sum() / (L * H * W)
            # Ego
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        return communication_masks, communication_rates


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x


class TrajectoryAwareWhere2comm(nn.Module):
    def __init__(self, args):
        super(TrajectoryAwareWhere2comm, self).__init__()
        
        # Initialize basic parameters first
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']
        self.fully = args.get('fully', False)  # Use get() with default value
        self.multi_scale = args.get('multi_scale', False)  # Use get() with default value
        
        if self.fully:
            print('constructing a fully connected communication graph')
        else:
            print('constructing a partially connected communication graph')

        # Initialize Communication module
        self.naive_communication = Communication(args['communication'])
        
        # Initialize Fusion modules
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttentionFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = AttentionFusion(args['in_channels'])
        
        # Add trajectory processing components
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(3, 64),  # [x, y, heading]
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        self.trajectory_attention = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x


    def process_trajectory(self, trajectory, H, W, device):
        """Process trajectory to create attention mask"""
        # Convert trajectory points to BEV coordinates
        traj_points = trajectory.clone()
        traj_points[..., 0] = traj_points[..., 0] / self.discrete_ratio
        traj_points[..., 1] = traj_points[..., 1] / self.discrete_ratio
        
        # Create attention mask - match confidence map size
        mask = torch.zeros((1, 1, H//4, W//4), device=device)  # Divide by 4 to match confidence map size
        sigma = 2.0
        kernel_size = 7
        
        # Encode trajectory features
        traj_features = self.trajectory_encoder(trajectory)  # [T, 128]
        traj_features = traj_features.mean(0, keepdim=True)  # [1, 128]
        
        # Create spatial attention - match confidence map size
        traj_features = traj_features.view(1, -1, 1, 1).expand(-1, -1, H//4, W//4)
        attention = self.trajectory_attention(traj_features)
        attention = torch.sigmoid(attention)
        
        # Scale coordinates to match reduced size
        scale_factor = 4  # Since confidence map is 4x smaller
        for point in traj_points:
            x = (point[0] / scale_factor).long()
            y = (point[1] / scale_factor).long()
            x = torch.clamp(x, 0, W//4-1)
            y = torch.clamp(y, 0, H//4-1)
            
            for i in range(max(0, y-kernel_size//2), min(H//4, y+kernel_size//2+1)):
                for j in range(max(0, x-kernel_size//2), min(W//4, x+kernel_size//2+1)):
                    dist = torch.sqrt(torch.tensor(float((i-y)**2 + (j-x)**2)))
                    mask[0, 0, i, j] += torch.exp(-dist**2 / (2*sigma**2))
        
        mask = F.normalize(mask, p=1, dim=(2,3))
        return mask * attention
    def forward(self, x, psm_single, record_len, pairwise_t_matrix, trajectory, backbone=None):
        """
        Modified forward pass incorporating trajectory information
        
        Args:
            x: Input features [N, C, H, W]
            psm_single: Detection head output
            record_len: Number of valid CAVs per batch 
            pairwise_t_matrix: Transformation matrices [B, L, L, 4, 4]
            trajectory: Trajectory data tensor [B, max_cav, T, 3]
            backbone: Optional backbone network for multi-scale
        """
        _, C, H, W = x.shape
        B = pairwise_t_matrix.shape[0]

        if self.multi_scale:
            ups = []

            for i in range(self.num_levels):
                x = backbone.blocks[i](x)

                if i == 0:
                    if self.fully:
                        communication_rates = torch.tensor(1).to(x.device)
                    else:
                        # Get basic confidence maps
                        batch_confidence_maps = self.regroup(psm_single, record_len)
                        
                        # Modify confidence maps with trajectory information
                        modified_maps = []
                        current_idx = 0
                        
                        for b in range(B):
                            conf_map = batch_confidence_maps[b]
                            num_cavs = conf_map.shape[0]  # Number of CAVs in this batch
                            
                            # Process each CAV's trajectory
                            batch_masks = []
                            for cav_idx in range(num_cavs):
                                # Get trajectory for this CAV
                                cav_traj = trajectory[b, cav_idx]  # [T, 3]
                                
                                if torch.any(cav_traj):  # If trajectory is not all zeros
                                    traj_mask = self.process_trajectory(cav_traj, H, W, x.device)
                                    modified_map = conf_map[cav_idx:cav_idx+1] * (1 + traj_mask)
                                else:
                                    modified_map = conf_map[cav_idx:cav_idx+1]
                                batch_masks.append(modified_map)
                            
                            # Combine all CAV masks for this batch
                            modified_map = torch.cat(batch_masks, dim=0)
                            modified_maps.append(modified_map)
                            
                        # Generate communication masks
                        communication_masks, communication_rates = self.naive_communication(modified_maps, B)
                        
                        if x.shape[-1] != communication_masks.shape[-1]:
                            communication_masks = F.interpolate(
                                communication_masks, 
                                size=(x.shape[-2], x.shape[-1]),
                                mode='bilinear', 
                                align_corners=False
                            )
                        x = x * communication_masks

                batch_node_features = self.regroup(x, record_len)
                x_fuse = []
                for b in range(B):
                    neighbor_feature = batch_node_features[b]
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)

                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)

        else:
            if self.fully:
                communication_rates = torch.tensor(1).to(x.device)
            else:
                batch_confidence_maps = self.regroup(psm_single, record_len)
                
                # Modify confidence maps with trajectory information
                modified_maps = []
                for b in range(B):
                    conf_map = batch_confidence_maps[b]
                    num_cavs = conf_map.shape[0]  # Number of CAVs in this batch
                    
                    # Process each CAV's trajectory
                    batch_masks = []
                    for cav_idx in range(num_cavs):
                        # Get trajectory for this CAV
                        cav_traj = trajectory[b, cav_idx]  # [T, 3]
                        
                        if torch.any(cav_traj):  # If trajectory is not all zeros
                            traj_mask = self.process_trajectory(cav_traj, H, W, x.device)
                            modified_map = conf_map[cav_idx:cav_idx+1] * (1 + traj_mask)
                        else:
                            modified_map = conf_map[cav_idx:cav_idx+1]
                        batch_masks.append(modified_map)
                    
                    modified_map = torch.cat(batch_masks, dim=0)
                    modified_maps.append(modified_map)
                
                communication_masks, communication_rates = self.naive_communication(modified_maps, B)
                x = x * communication_masks

            batch_node_features = self.regroup(x, record_len)
            x_fuse = []
            for b in range(B):
                neighbor_feature = batch_node_features[b]
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)

        return x_fuse, communication_rates
