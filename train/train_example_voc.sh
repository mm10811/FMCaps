#!/bin/bash
# FMCaps VOC Training Example Script

echo "=== FMCaps Capsule Network Training Examples ==="

# 1. Standard training with Capsule Network
echo "1. Training with Capsule Network..."
python voc_train_capsule.py \
    --config ../configs/voc_attn_reg.yaml \
    --work_dir experiment_fmcaps_voc \
    --crop_size 320 \
    --radius 8 \
    --pseudo_label_dir ../VOC2012/pseudo_labels_sgfr \
    --capsule_loss_weight 0.1 \
    --primary_caps_num 32 \
    --primary_caps_dim 8 \
    --num_routing 3

echo "Training completed"

# 2. Training without Capsule Network (baseline)
# echo "2. Training baseline without Capsule Network..."
# python voc_train_capsule.py \
#     --config ../configs/voc_attn_reg.yaml \
#     --work_dir experiment_baseline_voc \
#     --crop_size 320 \
#     --radius 8 \
#     --pseudo_label_dir ../VOC2012/pseudo_labels_sgfr \
#     --disable_capsule

echo "=== All training examples completed ==="
