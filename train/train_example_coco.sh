#!/bin/bash
# FMCaps COCO Training Example Script

echo "=== FMCaps Capsule Network Training Examples ==="

# 1. Training with Capsule Network on MS COCO 2014
echo "1. Training with Capsule Network..."
python coco_train_capsule.py \
    --config ../configs/coco_attn_reg.yaml \
    --work_dir experiment_fmcaps_coco \
    --crop_size 320 \
    --radius 8 \
    --pseudo_label_dir ../MSCOCO/pseudo_labels_sgfr \
    --capsule_loss_weight 0.1 \
    --primary_caps_num 32 \
    --primary_caps_dim 8 \
    --num_routing 3

echo "Training completed"

echo "=== All training examples completed ==="
