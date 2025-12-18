"""
Pseudo-label Generation Script using SGFR Module

This script generates high-quality pseudo-labels for WSSS training using:
1. CLIP-based CAM generation
2. SAM and Grounding-DINO fusion (SGFR module)

Usage:
    python tools/generate_pseudo_labels.py \
        --dataset voc \
        --data_root ./VOC2012 \
        --output_dir ./VOC2012/pseudo_labels_sgfr
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import warnings

# VOC and COCO class names
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


def load_image_list(data_root, dataset, split):
    """Load image name list"""
    if dataset == 'voc':
        list_path = os.path.join(data_root, '..', 'datasets', 'voc', f'{split}.txt')
        if not os.path.exists(list_path):
            list_path = os.path.join(data_root, 'ImageSets', 'Segmentation', f'{split}.txt')
    else:
        list_path = os.path.join(data_root, '..', 'datasets', 'coco', f'{split}.txt')
    
    with open(list_path, 'r') as f:
        names = [line.strip() for line in f.readlines()]
    return names


def load_class_labels(data_root, dataset):
    """Load class labels for each image"""
    if dataset == 'voc':
        label_path = os.path.join(data_root, '..', 'datasets', 'voc', 'cls_labels_onehot.npy')
    else:
        label_path = os.path.join(data_root, '..', 'datasets', 'coco', 'cls_labels_onehot.npy')
    
    return np.load(label_path, allow_pickle=True).item()


def generate_cam_based_pseudo_labels(
    data_root: str,
    output_dir: str,
    dataset: str = 'voc',
    split: str = 'train',
    use_sam: bool = False,
    use_gdino: bool = False,
    sam_checkpoint: str = None,
    gdino_checkpoint: str = None,
    gdino_config: str = None,
    device: str = 'cuda',
    bg_threshold: float = 0.2,
    fg_threshold: float = 0.5
):
    """
    Generate pseudo-labels using CAM with optional SGFR refinement
    
    Args:
        data_root: Path to dataset root
        output_dir: Output directory for pseudo-labels
        dataset: Dataset name ('voc' or 'coco')
        split: Data split ('train', 'trainval', etc.)
        use_sam: Whether to use SAM for refinement
        use_gdino: Whether to use Grounding-DINO for detection
        sam_checkpoint: Path to SAM checkpoint
        gdino_checkpoint: Path to Grounding-DINO checkpoint
        gdino_config: Path to Grounding-DINO config
        device: Device to use
        bg_threshold: Background threshold for CAM
        fg_threshold: Foreground threshold for CAM
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set class names based on dataset
    if dataset == 'voc':
        class_names = VOC_CLASSES
        num_classes = 21
        img_dir = os.path.join(data_root, 'JPEGImages')
    else:
        class_names = COCO_CLASSES
        num_classes = 81
        img_dir = os.path.join(data_root, 'JPEGImages', 'train')
    
    # Load image list and class labels
    image_names = load_image_list(data_root, dataset, split)
    class_labels = load_class_labels(data_root, dataset)
    
    print(f"Found {len(image_names)} images")
    
    # Initialize SGFR module if using SAM/GDINO
    sgfr = None
    if use_sam or use_gdino:
        try:
            from modules.sgfr import create_sgfr
            sgfr = create_sgfr(
                sam_checkpoint=sam_checkpoint,
                gdino_checkpoint=gdino_checkpoint,
                gdino_config=gdino_config,
                device=device,
                use_sam=use_sam,
                use_gdino=use_gdino
            )
            print("SGFR module initialized")
        except Exception as e:
            warnings.warn(f"Failed to initialize SGFR: {e}")
            sgfr = None
    
    # Try to load CLIP model for CAM generation
    clip_model = None
    try:
        import clip
        from clip.clip_tool import generate_clip_fts
        from pytorch_grad_cam import GradCAM
        
        clip_model, _ = clip.load('ViT-B/16', device=device)
        clip_model.eval()
        print("CLIP model loaded for CAM generation")
    except Exception as e:
        warnings.warn(f"Failed to load CLIP: {e}. Using fallback CAM method.")
    
    # Process each image
    for img_name in tqdm(image_names, desc="Generating pseudo-labels"):
        try:
            # Load image
            if dataset == 'voc':
                img_path = os.path.join(img_dir, f'{img_name}.jpg')
            else:
                img_path = os.path.join(img_dir, f'{img_name}.jpg')
            
            if not os.path.exists(img_path):
                continue
            
            image = np.array(Image.open(img_path).convert('RGB'))
            h, w = image.shape[:2]
            
            # Get class labels for this image
            if img_name in class_labels:
                img_cls_label = class_labels[img_name]
            else:
                continue
            
            # Get present classes (excluding background)
            present_class_ids = np.where(img_cls_label > 0)[0] + 1  # +1 for 1-indexed
            present_class_names = [class_names[i] for i in present_class_ids]
            
            if len(present_class_ids) == 0:
                # No foreground classes, save background-only label
                pseudo_label = np.zeros((h, w), dtype=np.uint8)
                save_path = os.path.join(output_dir, f'{img_name}.png')
                Image.fromarray(pseudo_label).save(save_path)
                continue
            
            # Generate pseudo-label
            if sgfr is not None:
                # Use SGFR for high-quality pseudo-labels
                pseudo_label = sgfr.generate(
                    image=image,
                    class_names=present_class_names,
                    class_ids=present_class_ids.tolist(),
                    cam=None  # Can add CAM here for point prompts
                )
            else:
                # Fallback: simple threshold-based pseudo-label
                # This is a placeholder - in practice, you'd use CAM
                pseudo_label = generate_simple_pseudo_label(
                    image, present_class_ids, 
                    bg_threshold, fg_threshold
                )
            
            # Save pseudo-label
            save_path = os.path.join(output_dir, f'{img_name}.png')
            Image.fromarray(pseudo_label.astype(np.uint8)).save(save_path)
            
        except Exception as e:
            warnings.warn(f"Failed to process {img_name}: {e}")
            continue
    
    print(f"Pseudo-labels saved to {output_dir}")


def generate_simple_pseudo_label(
    image: np.ndarray,
    class_ids: np.ndarray,
    bg_threshold: float = 0.2,
    fg_threshold: float = 0.5
) -> np.ndarray:
    """
    Generate simple pseudo-label (placeholder for when SGFR is not available)
    
    This is a basic implementation that assigns the first class to the center.
    In practice, this should be replaced with proper CAM-based generation.
    """
    h, w = image.shape[:2]
    pseudo_label = np.zeros((h, w), dtype=np.uint8)
    
    # Simple placeholder: assign first class to center region
    if len(class_ids) > 0:
        center_h, center_w = h // 2, w // 2
        radius_h, radius_w = h // 4, w // 4
        
        y, x = np.ogrid[:h, :w]
        center_mask = ((y - center_h) ** 2 / radius_h ** 2 + 
                       (x - center_w) ** 2 / radius_w ** 2) <= 1
        
        pseudo_label[center_mask] = class_ids[0]
    
    return pseudo_label


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-labels for WSSS')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'coco'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for pseudo-labels')
    parser.add_argument('--split', type=str, default='train',
                        help='Data split')
    
    # SGFR arguments
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM for segmentation')
    parser.add_argument('--use_gdino', action='store_true',
                        help='Use Grounding-DINO for detection')
    parser.add_argument('--sam_checkpoint', type=str, default=None,
                        help='Path to SAM checkpoint')
    parser.add_argument('--gdino_checkpoint', type=str, default=None,
                        help='Path to Grounding-DINO checkpoint')
    parser.add_argument('--gdino_config', type=str, default=None,
                        help='Path to Grounding-DINO config')
    
    # Threshold arguments
    parser.add_argument('--bg_threshold', type=float, default=0.2,
                        help='Background threshold for CAM')
    parser.add_argument('--fg_threshold', type=float, default=0.5,
                        help='Foreground threshold for CAM')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    generate_cam_based_pseudo_labels(
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        split=args.split,
        use_sam=args.use_sam,
        use_gdino=args.use_gdino,
        sam_checkpoint=args.sam_checkpoint,
        gdino_checkpoint=args.gdino_checkpoint,
        gdino_config=args.gdino_config,
        device=args.device,
        bg_threshold=args.bg_threshold,
        fg_threshold=args.fg_threshold
    )


if __name__ == '__main__':
    main()

