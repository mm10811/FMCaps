"""
Evaluation Script for FMCaps

Evaluates semantic segmentation performance using mIoU metric.

Usage:
    python tools/evaluate.py \
        --pred_dir results/predictions \
        --gt_dir VOC2012/SegmentationClass \
        --dataset voc
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_iou(pred, gt, num_classes, ignore_index=255):
    """
    Compute IoU for each class
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        num_classes: Number of classes
        ignore_index: Index to ignore in evaluation
        
    Returns:
        iou_per_class: IoU for each class
        mean_iou: Mean IoU
    """
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)
        
        # Ignore regions
        valid = (gt != ignore_index)
        pred_cls = pred_cls & valid
        gt_cls = gt_cls & valid
        
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = np.nan
        
        iou_per_class.append(iou)
    
    # Compute mean IoU (excluding NaN)
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    return iou_per_class, mean_iou


def evaluate_segmentation(
    pred_dir: str,
    gt_dir: str,
    dataset: str = 'voc',
    ignore_index: int = 255
):
    """
    Evaluate semantic segmentation predictions
    
    Args:
        pred_dir: Directory containing prediction masks
        gt_dir: Directory containing ground truth masks
        dataset: Dataset name ('voc' or 'coco')
        ignore_index: Index to ignore in evaluation
    """
    # Set number of classes
    if dataset == 'voc':
        num_classes = 21
        class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    else:
        num_classes = 81
        class_names = [f'class_{i}' for i in range(num_classes)]
    
    # Get list of prediction files
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.png')]
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Evaluate each prediction
    for pred_file in tqdm(pred_files, desc="Evaluating"):
        # Load prediction
        pred_path = os.path.join(pred_dir, pred_file)
        pred = np.array(Image.open(pred_path))
        
        # Load ground truth
        gt_path = os.path.join(gt_dir, pred_file)
        if not os.path.exists(gt_path):
            continue
        gt = np.array(Image.open(gt_path))
        
        # Update confusion matrix
        valid_mask = (gt != ignore_index)
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]
        
        for p, g in zip(pred_valid.flatten(), gt_valid.flatten()):
            if g < num_classes and p < num_classes:
                confusion_matrix[g, p] += 1
    
    # Compute IoU from confusion matrix
    iou_per_class = []
    for cls in range(num_classes):
        tp = confusion_matrix[cls, cls]
        fp = confusion_matrix[:, cls].sum() - tp
        fn = confusion_matrix[cls, :].sum() - tp
        
        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
        else:
            iou = np.nan
        
        iou_per_class.append(iou)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    print("\nPer-class IoU:")
    for cls in range(num_classes):
        if not np.isnan(iou_per_class[cls]):
            print(f"  {class_names[cls]:20s}: {iou_per_class[cls]*100:.2f}%")
    
    # Mean IoU
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    print(f"\n{'Mean IoU':20s}: {mean_iou*100:.2f}%")
    print("=" * 60)
    
    return iou_per_class, mean_iou


def main():
    parser = argparse.ArgumentParser(description='Evaluate semantic segmentation')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing prediction masks')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth masks')
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'coco'],
                        help='Dataset name')
    parser.add_argument('--ignore_index', type=int, default=255,
                        help='Index to ignore in evaluation')
    
    args = parser.parse_args()
    
    evaluate_segmentation(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        dataset=args.dataset,
        ignore_index=args.ignore_index
    )


if __name__ == '__main__':
    main()

