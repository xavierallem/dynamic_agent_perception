import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
import random


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
        raw_confidence_maps = []
        
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            # Store the raw confidence maps for potential use as node features
            raw_confidence_maps.append(communication_maps)
            
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
            # Ego always communicates
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
            
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        
        return communication_masks, communication_rates, raw_confidence_maps


class GraphConstruction:
    """
    Constructs a graph based on vehicle positions
    """
    def __init__(self, edge_threshold=10.0, fully_connected=False, min_connections=2):
        self.edge_threshold = edge_threshold
        self.fully_connected = fully_connected
        self.min_connections = min_connections
    
    def build_graph(self, node_features, pairwise_t_matrix):
        """
        Build a graph from vehicle features and their transformation matrices
        
        Args:
            node_features: List of features for each node
            pairwise_t_matrix: Transformation matrices (B, L, L, 4, 4)
        """
        num_vehicles = len(node_features)
        
        # Create edge list based on configuration
        edge_list = []
        
        if self.fully_connected:
            # Create fully connected graph (excluding self-loops)
            for i in range(num_vehicles):
                for j in range(num_vehicles):
                    if i != j:  # Skip self-loops
                        edge_list.append([i, j])
        else:
            # For simplicity, create a ring topology with connections to ego
            for i in range(num_vehicles):
                # Connect to previous node (with wrap-around)
                prev_node = (i - 1) % num_vehicles
                edge_list.append([i, prev_node])
                
                # Connect to next node (with wrap-around)
                next_node = (i + 1) % num_vehicles
                edge_list.append([i, next_node])
                
            # Always ensure the ego vehicle (idx 0) is connected to all other vehicles
            for i in range(1, num_vehicles):
                # Add connections both ways for the ego vehicle
                edge_list.append([0, i])
                edge_list.append([i, 0])
        
        # Remove duplicates
        edge_list = [list(t) for t in set(tuple(e) for e in edge_list)]
        
        # Convert to tensor and proper format for PyG
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, device=node_features[0].device).t()
        else:
            # Fallback - create self-loops
            edge_index = torch.tensor([[i, i] for i in range(num_vehicles)], device=node_features[0].device).t()
            
        return edge_index


class ConfidenceGNNFusion(nn.Module):
    """
    GNN fusion module that can use either regular features or confidence maps as node features
    """
    def __init__(self, feature_dim, hidden_dim=128, gnn_type='gat', num_layers=2, 
                 confidence_as_node_feature=False):
        super(ConfidenceGNNFusion, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.confidence_as_node_feature = confidence_as_node_feature
        
        # Feature encoder (for regular mode)
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Confidence feature encoder (for confidence-as-node-feature mode)
        if confidence_as_node_feature:
            self.confidence_encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)  # Global pooling to get one value per vehicle
            )
            
            # For confidence mode, we need a smaller GNN since inputs are just values
            gnn_in_dim = 32
            self.confidence_projection = nn.Linear(32, hidden_dim)
        else:
            gnn_in_dim = hidden_dim
        
        # Spatial feature processor
        self.spatial_processor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        if gnn_type == 'gcn':
            for i in range(num_layers):
                in_channels = gnn_in_dim if i == 0 else hidden_dim
                self.gnn_layers.append(GCNConv(in_channels, hidden_dim))
        elif gnn_type == 'gat':
            for i in range(num_layers):
                in_channels = gnn_in_dim if i == 0 else hidden_dim
                self.gnn_layers.append(GATConv(in_channels, hidden_dim, heads=4, concat=False))
        
        # Output projection
        self.output_projector = nn.Conv2d(hidden_dim, feature_dim, kernel_size=1)
    
    def forward(self, x, edge_index, confidence_maps=None, batch=None):
        """
        Args:
            x: Node features (N, C, H, W)
            edge_index: Graph connectivity (2, E)
            confidence_maps: Confidence maps (N, 1, H, W) - optional
            batch: Batch assignment (N,) - optional
        """
        batch_size = x.shape[0]
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        
        if self.confidence_as_node_feature and confidence_maps is not None:
            # Mode 1: Use confidence maps as node features for the GNN
            
            # Ensure confidence maps have the right shape
            if confidence_maps.shape[-2:] != (H, W):
                confidence_maps = F.interpolate(
                    confidence_maps,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Process confidence maps to extract node features
            conf_features = self.confidence_encoder(confidence_maps)  # (N, 32, 1, 1)
            conf_features = conf_features.view(batch_size, -1)  # (N, 32)
            
            # Process original features normally for later combination
            x_encoded = self.feature_encoder(x)  # (N, hidden_dim, H, W)
            x_processed = self.spatial_processor(x_encoded)  # (N, hidden_dim, H, W)
            
            # Apply GNN layers on confidence features
            x_gnn = conf_features
            for gnn_layer in self.gnn_layers:
                x_gnn = gnn_layer(x_gnn, edge_index)
                x_gnn = F.relu(x_gnn)
            
            # Project confidence GNN output to feature space
            x_gnn = self.confidence_projection(x_gnn)  # (N, hidden_dim)
            
            # Expand to spatial dimensions
            x_gnn_expanded = x_gnn.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            
            # Combine with processed features
            x_combined = x_processed * x_gnn_expanded  # Modulation approach
            
        else:
            # Mode 2: Standard approach - use features as node features
            
            # Encode input features
            x_encoded = self.feature_encoder(x)  # (N, hidden_dim, H, W)
            
            # Weight with confidence if available
            if confidence_maps is not None:
                if confidence_maps.shape[-2:] != (H, W):
                    confidence_maps = F.interpolate(
                        confidence_maps,
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    )
                x_encoded = x_encoded * confidence_maps  # Weight by confidence
            
            # Process spatial features
            x_processed = self.spatial_processor(x_encoded)  # (N, hidden_dim, H, W)
            
            # Global average pooling to get node features
            x_node = F.adaptive_avg_pool2d(x_processed, 1).squeeze(-1).squeeze(-1)  # (N, hidden_dim)
            
            # Apply GNN layers
            x_gnn = x_node
            for gnn_layer in self.gnn_layers:
                x_gnn = gnn_layer(x_gnn, edge_index)
                x_gnn = F.relu(x_gnn)
            
            # Expand node features back to spatial dimensions
            x_gnn_expanded = x_gnn.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            
            # Simple addition of features
            x_combined = x_gnn_expanded + x_processed
        
        # Project to output dimension
        x_out = self.output_projector(x_combined)
        
        return x_out


class Where2commGNN(nn.Module):
    """
    Where2comm with GNN that leverages confidence maps
    """
    def __init__(self, args):
        super(Where2commGNN, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']
        
        self.fully = args['fully']
        if self.fully:
            print('constructing a fully connected communication graph')
        else:
            print('constructing a partially connected communication graph')
        
        # Communication module for feature selection
        self.naive_communication = Communication(args['communication'])
        
        # Graph construction module
        self.graph_constructor = GraphConstruction(
            edge_threshold=args.get('edge_threshold', 10.0),
            fully_connected=self.fully,
            min_connections=args.get('min_connections', 2)
        )
        
        # Whether to use confidence maps as node features in the GNN
        self.confidence_as_node_feature = args.get('confidence_as_node_feature', False)
        if self.confidence_as_node_feature:
            print('using confidence maps as GNN node features')
        
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            
            # GNN modules for each scale
            self.gnn_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                gnn_network = ConfidenceGNNFusion(
                    feature_dim=num_filters[idx],
                    hidden_dim=args.get('gnn_hidden_dim', 128),
                    gnn_type=args.get('gnn_type', 'gat'),
                    num_layers=args.get('gnn_layers', 2),
                    confidence_as_node_feature=self.confidence_as_node_feature
                )
                self.gnn_modules.append(gnn_network)
        else:
            # Single GNN module
            self.gnn_modules = ConfidenceGNNFusion(
                feature_dim=args['in_channels'],
                hidden_dim=args.get('gnn_hidden_dim', 128),
                gnn_type=args.get('gnn_type', 'gat'),
                num_layers=args.get('gnn_layers', 2),
                confidence_as_node_feature=self.confidence_as_node_feature
            )
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
    def forward(self, x, psm_single, record_len, pairwise_t_matrix, backbone=None):
        """
        Fusion forwarding with confidence-enhanced GNN
        """
        _, C, H, W = x.shape
        B = pairwise_t_matrix.shape[0]
        
        if self.multi_scale:
            ups = []
            
            for i in range(self.num_levels):
                x = backbone.blocks[i](x)
                
                # 1. Get communication masks and confidence maps
                if i == 0:
                    if self.fully:
                        communication_rates = torch.tensor(1).to(x.device)
                        communication_masks = torch.ones((x.shape[0], 1, H, W), device=x.device)
                        raw_confidence_maps = [None] * B
                    else:
                        # Get communication masks and raw confidence maps
                        batch_confidence_maps = self.regroup(psm_single, record_len)
                        communication_masks, communication_rates, raw_confidence_maps = self.naive_communication(
                            batch_confidence_maps, B)
                        
                        # Resize masks if needed
                        if x.shape[-1] != communication_masks.shape[-1]:
                            communication_masks = F.interpolate(
                                communication_masks, 
                                size=(x.shape[-2], x.shape[-1]),
                                mode='bilinear', 
                                align_corners=False
                            )
                        
                        # Apply masks to features for communication efficiency
                        x = x * communication_masks
                        
                        # Log bandwidth usage if not training
                        if not self.training:
                            activated_elements = (x != 0).sum().item()
                            bandwidth = activated_elements * 4 / 1000000  # MB
                            print(f"\n🌐 Bandwidth: {bandwidth:.2f} MB")
                            

                
                # 2. Split features and confidence maps by batch
                batch_node_features = self.regroup(x, record_len)
                batch_confidence = self.regroup(communication_masks, record_len) if not self.fully else [None] * B
                
                # 3. GNN Fusion
                x_fuse = []
                for b in range(B):
                    # Get features for this batch
                    neighbor_features = batch_node_features[b]
                    neighbor_confidence = batch_confidence[b]
                    
                    # Build graph
                    edge_index = self.graph_constructor.build_graph(
                        neighbor_features,
                        pairwise_t_matrix[b:b+1]
                    )
                    
                    # Apply GNN with appropriate confidence maps
                    if i < len(self.gnn_modules):
                        if self.confidence_as_node_feature and raw_confidence_maps[b] is not None:
                            # If using confidence as node features, pass raw confidence maps
                            fused_features = self.gnn_modules[i](
                                neighbor_features,
                                edge_index,
                                raw_confidence_maps[b]
                            )
                        else:
                            # Otherwise use standard approach with mask-based weighting
                            fused_features = self.gnn_modules[i](
                                neighbor_features,
                                edge_index,
                                neighbor_confidence
                            )
                    else:
                        # Fallback
                        fused_features = neighbor_features
                    
                    # Take ego vehicle features
                    x_fuse.append(fused_features[0:1])
                
                x_fuse = torch.cat(x_fuse, dim=0)
                
                # 4. Deconv
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
            # Single scale processing
            
            # 1. Get communication masks and confidence maps
            if self.fully:
                communication_rates = torch.tensor(1).to(x.device)
                communication_masks = torch.ones((x.shape[0], 1, H, W), device=x.device)
                raw_confidence_maps = [None] * B
            else:
                # Get communication masks and raw confidence maps
                batch_confidence_maps = self.regroup(psm_single, record_len)
                communication_masks, communication_rates, raw_confidence_maps = self.naive_communication(
                    batch_confidence_maps, B)
                
                # Apply masks to features for communication efficiency
                x = x * communication_masks
                
                # Log bandwidth usage if not training
                if not self.training:
                    activated_elements = (x != 0).sum().item()
                    bandwidth = activated_elements * 4 / 1000000  # MB
                    print(f"\n🌐 Bandwidth: {bandwidth:.2f} MB")
                    

            # 2. Split features and confidence maps by batch
            batch_node_features = self.regroup(x, record_len)
            batch_confidence = self.regroup(communication_masks, record_len) if not self.fully else [None] * B
            
            # 3. GNN Fusion
            x_fuse = []
            for b in range(B):
                # Get features for this batch
                neighbor_features = batch_node_features[b]
                neighbor_confidence = batch_confidence[b]
                
                # Build graph
                edge_index = self.graph_constructor.build_graph(
                    neighbor_features,
                    pairwise_t_matrix[b:b+1]
                )
                
                # Apply GNN with appropriate confidence maps
                if self.confidence_as_node_feature and raw_confidence_maps[b] is not None:
                    # If using confidence as node features, pass raw confidence maps
                    fused_features = self.gnn_modules(
                        neighbor_features,
                        edge_index,
                        raw_confidence_maps[b]
                    )
                else:
                    # Otherwise use standard approach with mask-based weighting
                    fused_features = self.gnn_modules(
                        neighbor_features,
                        edge_index,
                        neighbor_confidence
                    )
                
                # Take ego vehicle features
                x_fuse.append(fused_features[0:1])
            
            x_fuse = torch.cat(x_fuse, dim=0)
        
        return x_fuse, communication_rates