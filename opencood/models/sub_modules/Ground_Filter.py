import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RANSACRegressor

class GroundFilter(nn.Module):
    def __init__(self, args):
        """
        Ground filtering module using RANSAC
        
        Args:
            args (dict): Configuration with keys:
                - ground_threshold: max distance from ground plane (default: 0.2)
                - min_points: minimum points for RANSAC (default: 1000)
                - ransac_thresh: RANSAC threshold (default: 0.1)
        """
        super(GroundFilter, self).__init__()
        self.ground_threshold = args.get('ground_threshold', 0.2)
        self.min_points = args.get('min_points', 1000)
        self.ransac_thresh = args.get('ransac_thresh', 0.1)
        
        self.ransac = RANSACRegressor(
            residual_threshold=self.ransac_thresh,
            random_state=0
        )
        
    def estimate_ground_plane(self, points):
        """
        Estimate ground plane using RANSAC
        
        Args:
            points (np.ndarray): (N, 3) point cloud
            
        Returns:
            tuple: (normal vector, plane offset)
        """
        if len(points) < self.min_points:
            return None, None
            
        X = points[:, [0, 1]]  # x, y coordinates
        y = points[:, 2]       # z coordinates
        
        try:
            self.ransac.fit(X, y)
            normal = np.array([
                self.ransac.estimator_.coef_[0],
                self.ransac.estimator_.coef_[1],
                -1
            ])
            d = -self.ransac.estimator_.intercept_
            return normal, d
        except:
            return None, None
            
    def get_ground_mask(self, points, normal, d):
        """
        Get binary mask for points near ground plane
        
        Args:
            points (np.ndarray): (N, 3) point cloud
            normal (np.ndarray): Ground plane normal vector
            d (float): Ground plane offset
            
        Returns:
            np.ndarray: Binary mask for ground points
        """
        if normal is None or d is None:
            return np.ones(len(points), dtype=bool)
            
        distances = np.abs(np.dot(points[:, :3], normal) + d) 
        distances /= np.linalg.norm(normal)
        return distances < self.ground_threshold
        
    def filter_features(self, features, points):
        if isinstance(points, list):
            points = torch.cat(points, dim=0).cpu().numpy()
        elif isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
            
        normal, d = self.estimate_ground_plane(points)
        ground_mask = self.get_ground_mask(points, normal, d)
        
        if isinstance(features, torch.Tensor):
            ground_mask = torch.from_numpy(ground_mask).to(features.device)
            # Match first dimension to features
            ground_mask = ground_mask[:features.shape[0]]
            ground_mask = ground_mask.view(-1, 1, 1).expand_as(features)
        
        return features * ground_mask
        
    def forward(self, features, points):
        """
        Forward pass
        
        Args:
            features (torch.Tensor): Input features
            points (torch.Tensor/np.ndarray): Point cloud
            
        Returns:
            torch.Tensor: Filtered features
        """
        return self.filter_features(features, points)