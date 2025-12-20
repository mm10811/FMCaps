#!/bin/bash
# FMCaps Testing Example Script

echo "=== FMCaps Evaluation Examples ==="

# 1. Evaluate on PASCAL VOC val set
echo "1. Evaluating on PASCAL VOC 2012 val set..."
python test_msc_flip_voc.py \
    --config ../configs/voc_attn_reg.yaml \
    --checkpoint ../checkpoints/fmcaps_voc_best.pth \
    --save_dir ../results/voc_val

# 2. Evaluate on MS COCO val set
echo "2. Evaluating on MS COCO 2014 val set..."
python test_msc_flip_coco.py \
    --config ../configs/coco_attn_reg.yaml \
    --checkpoint ../checkpoints/fmcaps_coco_best.pth \
    --save_dir ../results/coco_val

echo "=== Evaluation completed ==="
