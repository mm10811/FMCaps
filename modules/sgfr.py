"""
SGFR: SAM and Grounding-DINO Fusion Refinement Module

This module generates high-quality pseudo-labels by combining:
1. Grounding-DINO for open-set object detection with class name prompts
2. CAM-derived point prompts for comprehensive coverage
3. SAM for precise segmentation with both box and point prompts
4. Conflict resolution strategy for aggregating class-specific masks

Reference: 
    "Integrating Foundation Models with Capsule Networks for Enhanced 
    Weakly-Supervised Semantic Segmentation" - ESWA 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Optional, Union
import warnings


class SGFR(nn.Module):
    """
    SAM and Grounding-DINO Fusion Refinement Module
    
    Args:
        sam_checkpoint: Path to SAM checkpoint
        sam_model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        gdino_checkpoint: Path to Grounding-DINO checkpoint (optional)
        gdino_config: Path to Grounding-DINO config (optional)
        device: Device to run models on
        box_threshold: Confidence threshold for Grounding-DINO boxes
        text_threshold: Text confidence threshold for Grounding-DINO
        nms_threshold: NMS threshold for overlapping boxes
        cam_peak_threshold: Threshold for extracting CAM peaks as point prompts
        use_sam: Whether to use SAM for segmentation
        use_gdino: Whether to use Grounding-DINO for detection
    """
    
    def __init__(
        self,
        sam_checkpoint: str = None,
        sam_model_type: str = 'vit_h',
        gdino_checkpoint: str = None,
        gdino_config: str = None,
        device: str = 'cuda',
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        cam_peak_threshold: float = 0.5,
        use_sam: bool = True,
        use_gdino: bool = True
    ):
        super().__init__()
        
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.cam_peak_threshold = cam_peak_threshold
        self.use_sam = use_sam
        self.use_gdino = use_gdino
        
        # Initialize SAM
        self.sam_predictor = None
        if use_sam and sam_checkpoint:
            self._init_sam(sam_checkpoint, sam_model_type)
        
        # Initialize Grounding-DINO
        self.gdino_model = None
        if use_gdino and gdino_checkpoint:
            self._init_grounding_dino(gdino_checkpoint, gdino_config)
    
    def _init_sam(self, checkpoint: str, model_type: str):
        """Initialize Segment Anything Model"""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
            print(f"SAM ({model_type}) loaded successfully")
        except ImportError:
            warnings.warn("segment-anything not installed. SAM features disabled.")
            self.use_sam = False
        except Exception as e:
            warnings.warn(f"Failed to load SAM: {e}")
            self.use_sam = False
    
    def _init_grounding_dino(self, checkpoint: str, config: str):
        """Initialize Grounding-DINO model"""
        try:
            from groundingdino.util.inference import load_model
            self.gdino_model = load_model(config, checkpoint)
            self.gdino_model.to(self.device)
            print("Grounding-DINO loaded successfully")
        except ImportError:
            warnings.warn("groundingdino not installed. Grounding-DINO features disabled.")
            self.use_gdino = False
        except Exception as e:
            warnings.warn(f"Failed to load Grounding-DINO: {e}")
            self.use_gdino = False
    
    def detect_with_gdino(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Detect objects using Grounding-DINO
        
        Args:
            image: Input image (H, W, 3) in RGB format
            class_names: List of class names to detect
            
        Returns:
            Dictionary mapping class names to list of bounding boxes
        """
        if not self.use_gdino or self.gdino_model is None:
            return {name: [] for name in class_names}
        
        try:
            from groundingdino.util.inference import predict
            from groundingdino.util.utils import get_phrases_from_posmap
            
            # Create text prompt
            caption = " . ".join(class_names) + " ."
            
            # Run detection
            boxes, logits, phrases = predict(
                model=self.gdino_model,
                image=image,
                caption=caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )
            
            # Organize boxes by class
            result = {name: [] for name in class_names}
            for box, phrase in zip(boxes, phrases):
                # Match phrase to class name
                for class_name in class_names:
                    if class_name.lower() in phrase.lower():
                        # Convert normalized coords to absolute
                        h, w = image.shape[:2]
                        x1, y1, x2, y2 = box
                        box_abs = np.array([x1*w, y1*h, x2*w, y2*h])
                        result[class_name].append(box_abs)
                        break
            
            return result
            
        except Exception as e:
            warnings.warn(f"Grounding-DINO detection failed: {e}")
            return {name: [] for name in class_names}
    
    def extract_cam_points(
        self,
        cam: np.ndarray,
        threshold: float = None,
        max_points: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Extract salient point prompts from CAM local peaks
        
        Args:
            cam: Class Activation Map (H, W)
            threshold: Activation threshold
            max_points: Maximum number of points to extract
            
        Returns:
            List of (x, y) coordinates
        """
        if threshold is None:
            threshold = self.cam_peak_threshold
        
        # Normalize CAM
        cam = cam.astype(np.float32)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Find local maxima
        from scipy.ndimage import maximum_filter, label
        
        local_max = maximum_filter(cam, size=20) == cam
        threshold_mask = cam > threshold
        peaks = local_max & threshold_mask
        
        # Get peak coordinates
        y_coords, x_coords = np.where(peaks)
        
        if len(x_coords) == 0:
            # Fall back to global maximum
            y, x = np.unravel_index(cam.argmax(), cam.shape)
            return [(int(x), int(y))]
        
        # Sort by activation value and take top-k
        values = cam[y_coords, x_coords]
        sorted_indices = np.argsort(values)[::-1][:max_points]
        
        points = [(int(x_coords[i]), int(y_coords[i])) for i in sorted_indices]
        return points
    
    def segment_with_sam(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray] = None,
        points: List[Tuple[int, int]] = None,
        point_labels: List[int] = None
    ) -> List[np.ndarray]:
        """
        Segment objects using SAM with box and/or point prompts
        
        Args:
            image: Input image (H, W, 3) in RGB format
            boxes: List of bounding boxes [x1, y1, x2, y2]
            points: List of (x, y) point coordinates
            point_labels: Labels for points (1=foreground, 0=background)
            
        Returns:
            List of binary masks
        """
        if not self.use_sam or self.sam_predictor is None:
            return []
        
        try:
            # Set image
            self.sam_predictor.set_image(image)
            
            masks = []
            
            # Segment with boxes
            if boxes:
                for box in boxes:
                    box_array = np.array(box)
                    mask, _, _ = self.sam_predictor.predict(
                        box=box_array,
                        multimask_output=False
                    )
                    masks.append(mask[0])
            
            # Segment with points
            if points and not boxes:
                point_coords = np.array(points)
                if point_labels is None:
                    point_labels = np.ones(len(points))
                else:
                    point_labels = np.array(point_labels)
                
                mask, _, _ = self.sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False
                )
                masks.append(mask[0])
            
            return masks
            
        except Exception as e:
            warnings.warn(f"SAM segmentation failed: {e}")
            return []
    
    def resolve_conflicts(
        self,
        masks: Dict[int, List[np.ndarray]],
        h: int,
        w: int,
        ignore_index: int = 255
    ) -> np.ndarray:
        """
        Resolve conflicts between overlapping masks from different classes
        
        Strategy:
        - For overlapping regions, assign to the class with smaller mask area
          (assuming smaller objects are more likely to be occluded)
        - Background is assigned where no object mask exists
        
        Args:
            masks: Dictionary mapping class_id to list of masks
            h, w: Output dimensions
            ignore_index: Index for ignored regions
            
        Returns:
            Unified pseudo-label map (H, W)
        """
        # Initialize with background
        pseudo_label = np.zeros((h, w), dtype=np.uint8)
        
        # Track mask areas for conflict resolution
        area_map = np.full((h, w), np.inf)
        
        for class_id, class_masks in masks.items():
            if class_id == 0:  # Skip background
                continue
                
            for mask in class_masks:
                # Resize mask if needed
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), 
                                     interpolation=cv2.INTER_NEAREST)
                
                mask_bool = mask > 0
                mask_area = mask_bool.sum()
                
                # Update pixels where this mask has smaller area (conflict resolution)
                update_mask = mask_bool & (mask_area < area_map)
                pseudo_label[update_mask] = class_id
                area_map[update_mask] = mask_area
        
        return pseudo_label
    
    def generate(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        class_names: List[str],
        class_ids: List[int] = None,
        cam: np.ndarray = None,
        ignore_index: int = 255
    ) -> np.ndarray:
        """
        Generate high-quality pseudo-label for a single image
        
        Args:
            image: Input image
            class_names: List of class names present in the image
            class_ids: Corresponding class IDs (if None, uses 1-indexed)
            cam: Optional CAM for point prompt extraction (H, W, num_classes) or dict
            ignore_index: Index for ignored regions
            
        Returns:
            Pseudo-label map (H, W)
        """
        # Convert image to numpy RGB
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.ndim == 4:
                image = image[0]
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            # Denormalize if needed
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        
        if class_ids is None:
            class_ids = list(range(1, len(class_names) + 1))
        
        # Storage for all masks
        all_masks = {class_id: [] for class_id in class_ids}
        
        # Step 1: Detect with Grounding-DINO
        if self.use_gdino:
            detections = self.detect_with_gdino(image, class_names)
            
            # Step 2: Segment detected boxes with SAM
            for class_name, boxes in detections.items():
                class_idx = class_names.index(class_name)
                class_id = class_ids[class_idx]
                
                if boxes and self.use_sam:
                    masks = self.segment_with_sam(image, boxes=boxes)
                    all_masks[class_id].extend(masks)
        
        # Step 3: Extract point prompts from CAM and segment
        if cam is not None and self.use_sam:
            for i, class_name in enumerate(class_names):
                class_id = class_ids[i]
                
                # Get CAM for this class
                if isinstance(cam, dict):
                    class_cam = cam.get(class_name, cam.get(class_id))
                elif cam.ndim == 3:
                    class_cam = cam[:, :, i]
                else:
                    class_cam = cam
                
                if class_cam is not None:
                    # Extract point prompts
                    points = self.extract_cam_points(class_cam)
                    
                    if points:
                        # Segment with points
                        point_masks = self.segment_with_sam(
                            image, 
                            points=points,
                            point_labels=[1] * len(points)
                        )
                        all_masks[class_id].extend(point_masks)
        
        # Step 4: Resolve conflicts and create unified pseudo-label
        pseudo_label = self.resolve_conflicts(all_masks, h, w, ignore_index)
        
        return pseudo_label
    
    def generate_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        class_names_list: List[List[str]],
        class_ids_list: List[List[int]] = None,
        cams: List[np.ndarray] = None,
        ignore_index: int = 255
    ) -> List[np.ndarray]:
        """
        Generate pseudo-labels for a batch of images
        
        Args:
            images: List of input images
            class_names_list: List of class name lists for each image
            class_ids_list: List of class ID lists for each image
            cams: List of CAMs for each image
            ignore_index: Index for ignored regions
            
        Returns:
            List of pseudo-label maps
        """
        pseudo_labels = []
        
        for i, image in enumerate(images):
            class_names = class_names_list[i]
            class_ids = class_ids_list[i] if class_ids_list else None
            cam = cams[i] if cams else None
            
            pseudo_label = self.generate(
                image=image,
                class_names=class_names,
                class_ids=class_ids,
                cam=cam,
                ignore_index=ignore_index
            )
            pseudo_labels.append(pseudo_label)
        
        return pseudo_labels


class SimpleSGFR(nn.Module):
    """
    Simplified SGFR module without external dependencies
    Uses CAM refinement with morphological operations as fallback
    """
    
    def __init__(
        self,
        cam_threshold: float = 0.5,
        bg_threshold: float = 0.1,
        kernel_size: int = 5
    ):
        super().__init__()
        self.cam_threshold = cam_threshold
        self.bg_threshold = bg_threshold
        self.kernel_size = kernel_size
    
    def refine_cam(
        self,
        cam: np.ndarray,
        fg_threshold: float = None,
        bg_threshold: float = None
    ) -> np.ndarray:
        """
        Refine CAM using morphological operations
        
        Args:
            cam: Class Activation Map (H, W) or (H, W, C)
            fg_threshold: Foreground threshold
            bg_threshold: Background threshold
            
        Returns:
            Refined binary mask
        """
        if fg_threshold is None:
            fg_threshold = self.cam_threshold
        if bg_threshold is None:
            bg_threshold = self.bg_threshold
        
        # Normalize
        cam = cam.astype(np.float32)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Create initial mask
        fg_mask = (cam > fg_threshold).astype(np.uint8)
        
        # Morphological refinement
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.kernel_size, self.kernel_size)
        )
        
        # Close small holes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        return fg_mask
    
    def generate(
        self,
        cams: np.ndarray,
        class_ids: List[int],
        ignore_index: int = 255
    ) -> np.ndarray:
        """
        Generate pseudo-label from multi-class CAMs
        
        Args:
            cams: Multi-class CAMs (H, W, num_classes) or dict
            class_ids: List of class IDs
            ignore_index: Index for ignored/uncertain regions
            
        Returns:
            Pseudo-label map (H, W)
        """
        if isinstance(cams, dict):
            h, w = list(cams.values())[0].shape[:2]
        else:
            h, w = cams.shape[:2]
        
        # Initialize with background
        pseudo_label = np.zeros((h, w), dtype=np.uint8)
        confidence = np.zeros((h, w), dtype=np.float32)
        
        for i, class_id in enumerate(class_ids):
            if isinstance(cams, dict):
                cam = cams[class_id]
            else:
                cam = cams[:, :, i]
            
            # Refine CAM
            mask = self.refine_cam(cam)
            
            # Update pseudo-label where confidence is higher
            cam_norm = cam.astype(np.float32)
            if cam_norm.max() > cam_norm.min():
                cam_norm = (cam_norm - cam_norm.min()) / (cam_norm.max() - cam_norm.min())
            
            update_mask = (mask > 0) & (cam_norm > confidence)
            pseudo_label[update_mask] = class_id
            confidence[update_mask] = cam_norm[update_mask]
        
        # Mark low-confidence regions as ignore
        uncertain_mask = (confidence > self.bg_threshold) & (confidence < self.cam_threshold)
        # pseudo_label[uncertain_mask] = ignore_index  # Optional: mark uncertain regions
        
        return pseudo_label


def create_sgfr(
    sam_checkpoint: str = None,
    gdino_checkpoint: str = None,
    gdino_config: str = None,
    device: str = 'cuda',
    use_sam: bool = True,
    use_gdino: bool = True,
    **kwargs
) -> SGFR:
    """
    Factory function to create SGFR module
    Falls back to SimpleSGFR if dependencies are missing
    """
    if use_sam or use_gdino:
        try:
            return SGFR(
                sam_checkpoint=sam_checkpoint,
                gdino_checkpoint=gdino_checkpoint,
                gdino_config=gdino_config,
                device=device,
                use_sam=use_sam,
                use_gdino=use_gdino,
                **kwargs
            )
        except Exception as e:
            warnings.warn(f"Failed to create SGFR with SAM/GDINO: {e}. Using SimpleSGFR.")
    
    return SimpleSGFR(**kwargs)

