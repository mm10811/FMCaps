"""
Capsule Network Module for Weakly-Supervised Semantic Segmentation

This module implements the Capsule Network component of FMCaps framework,
which learns structured object representations through dynamic routing
and fuses predictions with the main decoder for enhanced segmentation.

Key Components:
    - PrimaryCapsule: Converts CNN features to capsule representations
    - RoutingCapsule: Dynamic routing for higher-level capsule computation
    - CapsuleSegmentationHead: Generates segmentation from capsule features
    - PlugAndPlayCapsuleModule: Main module for integration with WeCLIP

Reference:
    "Integrating Foundation Models with Capsule Networks for Enhanced 
    Weakly-Supervised Semantic Segmentation" - ESWA 2025
    
    "Dynamic Routing Between Capsules" - Sabour et al., NeurIPS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


def squash(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Squash activation function for capsule networks
    
    Normalizes the length of capsule vectors to be between 0 and 1,
    while preserving the direction of the vector.
    
    Args:
        tensor: Input tensor to squash
        dim: Dimension along which to compute the norm
        
    Returns:
        Squashed tensor with length in range (0, 1)
    """
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-8)


class PrimaryCapsule(nn.Module):
    """
    Primary Capsule Layer
    
    Converts CNN feature maps to primary capsule representations.
    Each capsule is a group of neurons that represent different properties
    of the same entity (e.g., pose, texture, deformation).
    
    Args:
        in_channels: Number of input channels from CNN
        out_channels: Number of capsule types (num_caps)
        dim_caps: Dimension of each capsule vector
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        
    Input:
        x: Feature map tensor (B, C, H, W)
        
    Output:
        Capsule tensor (B, num_caps, H, W, dim_caps)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dim_caps: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0
    ):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.num_caps = out_channels
        self.capsules = nn.Conv2d(
            in_channels, 
            out_channels * dim_caps, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        capsules = self.capsules(x)
        capsules = capsules.view(
            batch_size, self.num_caps, self.dim_caps, 
            capsules.size(2), capsules.size(3)
        )
        # Rearrange to (B, num_caps, H, W, dim_caps)
        capsules = capsules.permute(0, 1, 3, 4, 2).contiguous()
        return squash(capsules, dim=-1)


class RoutingCapsule(nn.Module):
    """
    Dynamic Routing Capsule Layer
    
    Implements the dynamic routing algorithm between capsule layers.
    This layer learns to route information from lower-level capsules
    to higher-level capsules based on agreement between predictions.
    
    Args:
        in_caps: Number of input capsule types
        in_dim: Dimension of input capsules
        out_caps: Number of output capsule types
        out_dim: Dimension of output capsules
        num_routing: Number of routing iterations
        
    Input:
        x: Input capsules (B, in_caps, in_dim)
        
    Output:
        Output capsules (B, out_caps, out_dim)
        
    Reference:
        "Dynamic Routing Between Capsules" - Sabour et al., NeurIPS 2017
    """
    
    def __init__(
        self, 
        in_caps: int, 
        in_dim: int, 
        out_caps: int, 
        out_dim: int, 
        num_routing: int = 3
    ):
        super(RoutingCapsule, self).__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing
        
        # Routing weight matrix: [out_caps, in_caps, in_dim, out_dim]
        self.route_weights = nn.Parameter(
            torch.randn(out_caps, in_caps, in_dim, out_dim) * 0.01
        )
        
    def squash(self, tensor, dim=-1):
        """Squash激活函数"""
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)
        
    def forward(self, x):
        """
        动态路由算法
        Args:
            x: [batch_size, in_caps, in_dim]
        Returns:
            outputs: [batch_size, out_caps, out_dim]
        """
        batch_size = x.size(0)
        
        # 计算预测向量 priors
        # x: [batch_size, in_caps, in_dim]
        # route_weights: [out_caps, in_caps, in_dim, out_dim]
        
        # 扩展维度进行矩阵乘法
        x_expanded = x[None, :, :, None, :]  # [1, batch_size, in_caps, 1, in_dim]
        route_weights_expanded = self.route_weights[:, None, :, :, :]  # [out_caps, 1, in_caps, in_dim, out_dim]
        
        # 计算预测向量: [out_caps, batch_size, in_caps, out_dim]
        priors = (x_expanded @ route_weights_expanded).squeeze(-2)
        
        # 初始化路由logits: [out_caps, batch_size, in_caps]
        logits = torch.zeros(self.out_caps, batch_size, self.in_caps, device=x.device)
        
        # 动态路由迭代
        for i in range(self.num_routing):
            # 计算路由概率: [out_caps, batch_size, in_caps]
            probs = F.softmax(logits, dim=2)
            
            # 计算输出胶囊: [out_caps, batch_size, out_dim]
            outputs = self.squash((probs.unsqueeze(-1) * priors).sum(dim=2))
            
            # 更新路由logits (除了最后一次迭代)
            if i != self.num_routing - 1:
                # 计算一致性: [out_caps, batch_size, in_caps]
                delta_logits = (priors * outputs.unsqueeze(2)).sum(dim=-1)
                logits = logits + delta_logits
        
        # 转换输出维度: [batch_size, out_caps, out_dim]
        return outputs.transpose(0, 1)


class CapsuleSegmentationHead(nn.Module):
    """胶囊网络分割头"""
    def __init__(self, in_caps, in_dim, num_classes, feature_size):
        super(CapsuleSegmentationHead, self).__init__()
        self.num_classes = num_classes
        self.feature_size = feature_size
        
        self.class_capsules = RoutingCapsule(
            in_caps=in_caps, in_dim=in_dim, out_caps=num_classes, out_dim=16, num_routing=3
        )
        
        # 动态计算重构网络的输入维度
        reconstruction_input_dim = num_classes * 16
        hidden_dim = max(128, min(512, reconstruction_input_dim * 2))
        
        self.reconstruction = nn.Sequential(
            nn.Linear(reconstruction_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_size * feature_size),
            nn.ReLU(inplace=True)
        )
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
    def forward(self, capsules):
        batch_size, in_caps, height, width, in_dim = capsules.shape
        
        # 将空间胶囊转换为全局胶囊 - 使用平均池化减少空间维度
        capsules_pooled = capsules.mean(dim=(2, 3))  # [batch_size, in_caps, in_dim]
        class_caps = self.class_capsules(capsules_pooled)
        
        class_caps_flat = class_caps.reshape(batch_size, -1)
        spatial_features = self.reconstruction(class_caps_flat)
        spatial_features = spatial_features.view(batch_size, 1, self.feature_size, self.feature_size)
        spatial_features = F.interpolate(spatial_features, size=(height, width), 
                                       mode='bilinear', align_corners=False)
        seg_logits = self.seg_head(spatial_features)
        
        return seg_logits, class_caps


class CapsuleLoss(nn.Module):
    """胶囊网络损失函数"""
    def __init__(self, num_classes, margin_loss_weight=1.0, reconstruction_loss_weight=0.0005):
        super(CapsuleLoss, self).__init__()
        self.num_classes = num_classes
        self.margin_loss_weight = margin_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        
    def margin_loss(self, class_caps, labels):
        batch_size = class_caps.size(0)
        caps_lengths = torch.sqrt((class_caps ** 2).sum(dim=2))
        
        # 确保标签是长整型，并处理ignore_index
        labels = labels.long()
        
        # 将ignore_index (通常是255) 替换为0，避免one_hot编码错误
        ignore_mask = (labels == 255)
        labels_clean = labels.clone()
        labels_clean[ignore_mask] = 0
        
        labels_flat = labels_clean.view(-1)
        labels_onehot = F.one_hot(labels_flat, num_classes=self.num_classes).float()
        labels_onehot = labels_onehot.view(batch_size, -1, self.num_classes)
        
        # 获取图像级别的标签（每个类别是否存在）
        image_labels = labels_onehot.max(dim=1)[0]
        
        # 对于ignore区域，将对应的图像级别标签设为0
        ignore_mask_flat = ignore_mask.view(batch_size, -1)
        for i in range(batch_size):
            if ignore_mask_flat[i].any():
                # 如果该batch中有ignore像素，重新计算图像级别标签
                valid_pixels = labels_onehot[i][~ignore_mask_flat[i]]
                if len(valid_pixels) > 0:
                    image_labels[i] = valid_pixels.max(dim=0)[0]
                else:
                    image_labels[i] = 0
        
        m_plus, m_minus, lambda_val = 0.9, 0.1, 0.5
        left = F.relu(m_plus - caps_lengths) ** 2
        right = F.relu(caps_lengths - m_minus) ** 2
        margin_loss = image_labels * left + lambda_val * (1 - image_labels) * right
        
        return margin_loss.sum(dim=1).mean()
    
    def forward(self, class_caps, labels):
        margin_loss = self.margin_loss(class_caps, labels)
        total_loss = self.margin_loss_weight * margin_loss
        return total_loss, {'margin_loss': margin_loss.item(), 'total_capsule_loss': total_loss.item()}


class PlugAndPlayCapsuleModule(nn.Module):
    """即插即用的胶囊网络模块"""
    def __init__(self, in_channels, num_classes, feature_size=20, primary_caps_num=32,
                 primary_caps_dim=8, class_caps_dim=16, num_routing=3,
                 enable_segmentation=True, enable_feature_enhancement=True):
        super(PlugAndPlayCapsuleModule, self).__init__()
        
        self.enable_segmentation = enable_segmentation
        self.enable_feature_enhancement = enable_feature_enhancement
        self.num_classes = num_classes

        #基础胶囊层，卷积转为胶囊，输出通道为输出维度*胶囊数
        self.primary_caps = PrimaryCapsule(
            in_channels=in_channels, out_channels=primary_caps_num,
            dim_caps=primary_caps_dim, kernel_size=3, stride=1, padding=1
        )

        #分割胶囊
        if self.enable_segmentation:
            self.seg_head = CapsuleSegmentationHead(
                in_caps=primary_caps_num, in_dim=primary_caps_dim,
                num_classes=num_classes, feature_size=feature_size
            )

        #特征增强
        if self.enable_feature_enhancement:
            self.feature_enhancement = nn.Sequential(
                nn.Conv2d(in_channels + num_classes, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        
        self.loss_fn = CapsuleLoss(num_classes=num_classes)
        
    def forward(self, features, return_enhanced_features=False):
        outputs = {}
        primary_caps = self.primary_caps(features)
        outputs['primary_caps'] = primary_caps
        
        if self.enable_segmentation:
            seg_logits, class_caps = self.seg_head(primary_caps)
            outputs['seg_logits'] = seg_logits
            outputs['class_caps'] = class_caps
        
        if self.enable_feature_enhancement and self.enable_segmentation:
            enhanced_features = torch.cat([features, seg_logits], dim=1)
            enhanced_features = self.feature_enhancement(enhanced_features)
            outputs['enhanced_features'] = enhanced_features
            
            if return_enhanced_features:
                return outputs, enhanced_features
        
        return outputs
    
    def compute_loss(self, outputs, labels):
        if 'class_caps' not in outputs:
            return torch.tensor(0.0, device=labels.device), {}
        
        class_caps = outputs['class_caps']
        loss, loss_dict = self.loss_fn(class_caps, labels)
        return loss, loss_dict


def create_capsule_module(in_channels, num_classes, feature_size=20, primary_caps_num=32,
                         primary_caps_dim=8, class_caps_dim=16, num_routing=3,
                         enable_segmentation=True, enable_feature_enhancement=True):
    """创建即插即用胶囊模块的工厂函数"""
    return PlugAndPlayCapsuleModule(
        in_channels=in_channels, num_classes=num_classes, feature_size=feature_size,
        primary_caps_num=primary_caps_num, primary_caps_dim=primary_caps_dim,
        class_caps_dim=class_caps_dim, num_routing=num_routing,
        enable_segmentation=enable_segmentation, enable_feature_enhancement=enable_feature_enhancement
    ) 