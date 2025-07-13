# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from opencood.models.sub_modules.bandwidth_util import save_to_csv,save_transmission_pipeline_features


class FlashScaledDotProductAttention(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(FlashScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)  # Match name with original
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim  # Match name with original
        
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        context = torch.bmm(attn, value)  # Match name with original
        
        return context  # Only return context to match original



class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x, record_len):
        if not self.training:
            original_features = x.clone()
            original_bandwidth = (original_features != 0).sum().item() * 4 / 1000000
            save_to_csv(original_bandwidth)
        split_x = self.regroup(x, record_len)
        C, W, H = split_x[0].shape[1:]
        out = []
        for xx in split_x:
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(xx, xx, xx)
            h = h.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...]
            out.append(h)
        return torch.stack(out)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
# class MoEScaledDotProductAttention(nn.Module):

#     def __init__(self, dim, num_experts=3):
#         super(MoEScaledDotProductAttention, self).__init__()
#         self.dim = dim
#         self.num_experts = num_experts
        
#         # Create multiple experts, each being a standard scaled dot product attention
#         self.experts = nn.ModuleList([
#             ScaledDotProductAttention(dim) for _ in range(num_experts)
#         ])
        
#         self.router = nn.Sequential(
#             nn.Linear(dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_experts),
#             nn.Softmax(dim=-1)
#         )
    
#     def forward(self, query, key, value):

#         H_W, cav_num, C = query.shape

#         pooled_query = query.mean(dim=1)  # [H*W, dim]
#         routing_weights = self.router(pooled_query)  # [H*W, num_experts]
        
#         # Apply each expert
#         expert_outputs = []
#         for i in range(self.num_experts):
           
#             expert_output = self.experts[i](query, key, value) 
#             expert_outputs.append(expert_output)
        
      
#         stacked_outputs = torch.stack(expert_outputs, dim=1)
        

#         routing_weights = routing_weights.view(H_W, self.num_experts, 1, 1)  
        
#         # Weighted sum of expert outputs
#         combined_output = (stacked_outputs * routing_weights).sum(dim=1) 
        
#         return combined_output
class MoEScaledDotProductAttention(nn.Module):
    def __init__(self, dim, num_experts=3):
        super(MoEScaledDotProductAttention, self).__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ScaledDotProductAttention(dim) for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
            #nn.Softmax(dim=-1)
        )
        with torch.no_grad():
              self.router[-1].bias.uniform_(-0.02, 0.02) # Add small random bias

    def forward(self, query, key, value):
        H_W, cav_num, C = query.shape
        pooled_query = query.mean(dim=1)

        # --- CORRECTED: Apply Softmax ONCE after getting logits ---
        logits = self.router(pooled_query) # Get raw scores (logits)
        routing_weights = F.softmax(logits, dim=-1) # Apply softmax HERE
        # --- END CORRECTION ---

        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = self.experts[i](query, key, value)
            expert_outputs.append(expert_output)
        stacked_outputs = torch.stack(expert_outputs, dim=1)

        routing_weights_reshaped = routing_weights.view(H_W, self.num_experts, 1, 1)
        combined_output = (stacked_outputs * routing_weights_reshaped).sum(dim=1)

        return combined_output, routing_weights, expert_outputs  # Return weights