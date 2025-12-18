"""
SGAE: SAM-Guided Affinity Enhancement Module

This module leverages SAM's structural priors to create structurally sound 
affinity targets that guide the decoder to learn pixel-wise feature similarities.

Key features:
1. Uses SAM's class-agnostic segmentation proposals as structural priors
2. Generates high-quality affinity targets based on segment boundaries
3. Guides the network to learn better pixel-wise relationships

Reference: 
    "Integrating Foundation Models with Capsule Networks for Enhanced 
    Weakly-Supervised Semantic Segmentation" - ESWA 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Union
import warnings


class SGAE(nn.Module):
    """
    SAM-Guided Affinity Enhancement Module
    
    Uses SAM's segment proposals to create structurally-aware affinity targets
    for training the segmentation network.
    
    Args:
        sam_checkpoint: Path to SAM checkpoint
        sam_model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        device: Device to run model on
        points_per_side: Number of points to sample per side for automatic mask generation
        pred_iou_thresh: Predicted IoU threshold for filtering masks
        stability_score_thresh: Stability score threshold for filtering masks
        min_mask_region_area: Minimum mask region area
    """
    
    def __init__(
        self,
        sam_checkpoint: str = None,
        sam_model_type: str = 'vit_h',
        device: str = 'cuda',
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100
    ):
        super().__init__()
        
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        
        self.sam_model = None
        self.mask_generator = None
        
        if sam_checkpoint:
            self._init_sam(sam_checkpoint, sam_model_type)
    
    def _init_sam(self, checkpoint: str, model_type: str):
        """Initialize SAM for automatic mask generation"""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(self.device)
            self.sam_model = sam
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=self.min_mask_region_area
            )
            print(f"SAM ({model_type}) initialized for SGAE")
            
        except ImportError:
            warnings.warn("segment-anything not installed. SGAE will use fallback method.")
        except Exception as e:
            warnings.warn(f"Failed to initialize SAM for SGAE: {e}")
    
    def generate_sam_segments(
        self,
        image: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generate class-agnostic segments using SAM
        
        Args:
            image: Input image (H, W, 3) in RGB format
            
        Returns:
            List of binary segment masks
        """
        if self.mask_generator is None:
            return []
        
        try:
            masks = self.mask_generator.generate(image)
            return [mask['segmentation'] for mask in masks]
        except Exception as e:
            warnings.warn(f"SAM segment generation failed: {e}")
            return []
    
    def compute_segment_affinity(
        self,
        segments: List[np.ndarray],
        size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Compute affinity matrix based on segment membership
        
        Pixels within the same segment have high affinity (1),
        pixels in different segments have low affinity (0).
        
        Args:
            segments: List of binary segment masks
            size: Output size (H, W)
            
        Returns:
            Affinity matrix (H*W, H*W)
        """
        h, w = size
        n_pixels = h * w
        
        # Initialize affinity matrix
        affinity = np.zeros((n_pixels, n_pixels), dtype=np.float32)
        
        for segment in segments:
            # Resize segment to target size
            if segment.shape != size:
                import cv2
                segment = cv2.resize(
                    segment.astype(np.uint8), 
                    (w, h), 
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Flatten segment
            segment_flat = segment.flatten().astype(bool)
            
            # Set affinity for pixels in this segment
            # Using outer product: affinity[i,j] = 1 if both i,j in segment
            indices = np.where(segment_flat)[0]
            if len(indices) > 0:
                affinity[np.ix_(indices, indices)] = 1.0
        
        return affinity
    
    def compute_boundary_affinity(
        self,
        segments: List[np.ndarray],
        size: Tuple[int, int],
        boundary_width: int = 3
    ) -> np.ndarray:
        """
        Compute affinity with boundary awareness
        
        Near-boundary pixels have uncertain affinity to encourage
        the network to learn precise boundaries.
        
        Args:
            segments: List of binary segment masks
            size: Output size (H, W)
            boundary_width: Width of boundary region
            
        Returns:
            Affinity matrix with boundary-aware weights
        """
        import cv2
        
        h, w = size
        
        # Compute combined boundary map
        boundary_map = np.zeros((h, w), dtype=np.float32)
        
        for segment in segments:
            if segment.shape != (h, w):
                segment = cv2.resize(
                    segment.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Detect boundaries using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(segment.astype(np.uint8), kernel, iterations=1)
            eroded = cv2.erode(segment.astype(np.uint8), kernel, iterations=1)
            boundary = dilated - eroded
            
            # Expand boundary region
            if boundary_width > 1:
                boundary = cv2.dilate(boundary, kernel, iterations=boundary_width-1)
            
            boundary_map = np.maximum(boundary_map, boundary.astype(np.float32))
        
        # Compute segment-based affinity
        segment_affinity = self.compute_segment_affinity(segments, size)
        
        # Reduce affinity weight near boundaries
        boundary_flat = boundary_map.flatten()
        boundary_weight = 1.0 - 0.5 * boundary_flat  # 0.5-1.0 range
        
        # Apply boundary weights
        weight_matrix = np.outer(boundary_weight, boundary_weight)
        weighted_affinity = segment_affinity * weight_matrix
        
        return weighted_affinity
    
    def compute_affinity_target(
        self,
        image: Union[np.ndarray, torch.Tensor],
        feature_size: Tuple[int, int],
        use_boundary_aware: bool = True,
        radius: int = 8
    ) -> torch.Tensor:
        """
        Compute affinity target for training
        
        Args:
            image: Input image
            feature_size: Size of feature map (H, W)
            use_boundary_aware: Whether to use boundary-aware affinity
            radius: Radius for local affinity (pixels beyond this distance are masked)
            
        Returns:
            Affinity target tensor (H*W, H*W)
        """
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.ndim == 4:
                image = image[0]
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
        
        # Generate SAM segments
        segments = self.generate_sam_segments(image)
        
        if not segments:
            # Fallback: uniform affinity
            h, w = feature_size
            return torch.ones(h * w, h * w)
        
        # Compute affinity
        if use_boundary_aware:
            affinity = self.compute_boundary_affinity(segments, feature_size)
        else:
            affinity = self.compute_segment_affinity(segments, feature_size)
        
        # Apply local radius mask
        if radius > 0:
            h, w = feature_size
            mask = self._create_radius_mask(h, w, radius)
            affinity = affinity * mask
        
        return torch.from_numpy(affinity).float()
    
    def _create_radius_mask(
        self,
        h: int,
        w: int,
        radius: int
    ) -> np.ndarray:
        """Create mask that only considers pixels within radius"""
        n = h * w
        mask = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            i_h, i_w = i // w, i % w
            
            for j in range(n):
                j_h, j_w = j // w, j % w
                
                if abs(i_h - j_h) <= radius and abs(i_w - j_w) <= radius:
                    mask[i, j] = 1.0
        
        return mask
    
    def forward(
        self,
        images: torch.Tensor,
        feature_size: Tuple[int, int] = None
    ) -> torch.Tensor:
        """
        Forward pass to compute affinity targets for a batch
        
        Args:
            images: Input images (B, C, H, W)
            feature_size: Target feature size
            
        Returns:
            Affinity targets (B, H*W, H*W)
        """
        batch_size = images.shape[0]
        
        if feature_size is None:
            feature_size = (images.shape[2] // 16, images.shape[3] // 16)
        
        affinity_targets = []
        
        for i in range(batch_size):
            affinity = self.compute_affinity_target(
                images[i],
                feature_size
            )
            affinity_targets.append(affinity)
        
        return torch.stack(affinity_targets, dim=0)


class SimpleAffinity(nn.Module):
    """
    Simple affinity computation without SAM dependency
    Uses color/feature similarity for affinity estimation
    """
    
    def __init__(
        self,
        sigma_color: float = 0.1,
        sigma_spatial: float = 5.0,
        radius: int = 8
    ):
        super().__init__()
        self.sigma_color = sigma_color
        self.sigma_spatial = sigma_spatial
        self.radius = radius
    
    def compute_color_affinity(
        self,
        image: torch.Tensor,
        feature_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Compute affinity based on color similarity
        
        Args:
            image: Input image (C, H, W)
            feature_size: Target size (h, w)
            
        Returns:
            Affinity matrix (h*w, h*w)
        """
        h, w = feature_size
        
        # Resize image to feature size
        image_resized = F.interpolate(
            image.unsqueeze(0),
            size=feature_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Flatten to (h*w, c)
        c = image_resized.shape[0]
        features = image_resized.view(c, -1).t()  # (h*w, c)
        
        # Compute pairwise color distance
        # Using broadcasting: (h*w, 1, c) - (1, h*w, c)
        diff = features.unsqueeze(1) - features.unsqueeze(0)
        color_dist = (diff ** 2).sum(dim=2)
        
        # Convert to affinity
        color_affinity = torch.exp(-color_dist / (2 * self.sigma_color ** 2))
        
        return color_affinity
    
    def compute_spatial_affinity(
        self,
        feature_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Compute affinity based on spatial proximity
        
        Args:
            feature_size: Feature size (h, w)
            
        Returns:
            Spatial affinity matrix (h*w, h*w)
        """
        h, w = feature_size
        n = h * w
        
        # Create coordinate grids
        y_coords = torch.arange(h).float().view(-1, 1).expand(h, w).reshape(-1)
        x_coords = torch.arange(w).float().view(1, -1).expand(h, w).reshape(-1)
        
        # Compute pairwise spatial distance
        y_diff = y_coords.unsqueeze(1) - y_coords.unsqueeze(0)
        x_diff = x_coords.unsqueeze(1) - x_coords.unsqueeze(0)
        spatial_dist = y_diff ** 2 + x_diff ** 2
        
        # Convert to affinity
        spatial_affinity = torch.exp(-spatial_dist / (2 * self.sigma_spatial ** 2))
        
        # Apply radius mask
        radius_mask = (torch.abs(y_diff) <= self.radius) & (torch.abs(x_diff) <= self.radius)
        spatial_affinity = spatial_affinity * radius_mask.float()
        
        return spatial_affinity
    
    def forward(
        self,
        images: torch.Tensor,
        feature_size: Tuple[int, int] = None
    ) -> torch.Tensor:
        """
        Compute affinity targets for a batch
        
        Args:
            images: Input images (B, C, H, W)
            feature_size: Target feature size
            
        Returns:
            Affinity targets (B, H*W, H*W)
        """
        if feature_size is None:
            feature_size = (images.shape[2] // 16, images.shape[3] // 16)
        
        batch_size = images.shape[0]
        device = images.device
        
        # Compute spatial affinity (same for all images)
        spatial_affinity = self.compute_spatial_affinity(feature_size).to(device)
        
        affinity_targets = []
        
        for i in range(batch_size):
            # Compute color affinity
            color_affinity = self.compute_color_affinity(images[i], feature_size)
            
            # Combine
            affinity = color_affinity * spatial_affinity
            affinity_targets.append(affinity)
        
        return torch.stack(affinity_targets, dim=0)


def create_sgae(
    sam_checkpoint: str = None,
    sam_model_type: str = 'vit_h',
    device: str = 'cuda',
    use_sam: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create SGAE module
    Falls back to SimpleAffinity if SAM is not available
    """
    if use_sam and sam_checkpoint:
        try:
            return SGAE(
                sam_checkpoint=sam_checkpoint,
                sam_model_type=sam_model_type,
                device=device,
                **kwargs
            )
        except Exception as e:
            warnings.warn(f"Failed to create SGAE with SAM: {e}. Using SimpleAffinity.")
    
    return SimpleAffinity(**kwargs)


class AffinityLoss(nn.Module):
    """
    Affinity loss for training with SGAE-generated targets
    
    Args:
        loss_type: Type of loss ('bce', 'mse', 'focal')
        focal_gamma: Gamma parameter for focal loss
    """
    
    def __init__(
        self,
        loss_type: str = 'bce',
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
    
    def forward(
        self,
        pred_affinity: torch.Tensor,
        target_affinity: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute affinity loss
        
        Args:
            pred_affinity: Predicted affinity (B, N, N)
            target_affinity: Target affinity (B, N, N)
            mask: Optional mask for valid positions
            
        Returns:
            Loss value
        """
        if mask is not None:
            pred_affinity = pred_affinity * mask
            target_affinity = target_affinity * mask
        
        if self.loss_type == 'bce':
            loss = F.binary_cross_entropy(
                pred_affinity.sigmoid(),
                target_affinity,
                reduction='mean'
            )
        elif self.loss_type == 'mse':
            loss = F.mse_loss(pred_affinity, target_affinity)
        elif self.loss_type == 'focal':
            bce = F.binary_cross_entropy_with_logits(
                pred_affinity,
                target_affinity,
                reduction='none'
            )
            pt = torch.exp(-bce)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss = (focal_weight * bce).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss

