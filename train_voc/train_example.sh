#!/bin/bash

# WeCLIP 胶囊网络训练示例脚本

echo "=== WeCLIP 胶囊网络训练示例 ==="

# 1. 标准WeCLIP训练（不使用胶囊网络）
echo "1. 标准WeCLIP训练..."
python train_voc_pseudo_capsule.py \
    --config ../configs/voc_attn_reg.yaml \
    --work_dir experiment_standard \
    --crop_size 320 \
    --radius 8 \
    --pseudo_label_dir ../VOC2012/pesudolabels_aug \
    --capsule_loss_weight 0.0

echo "标准训练完成"

# 2. 启用胶囊网络的训练
echo "2. 胶囊网络增强训练..."
python train_voc_pseudo_capsule.py \
    --config ../configs/voc_attn_reg.yaml \
    --work_dir experiment_capsule \
    --crop_size 320 \
    --radius 8 \
    --pseudo_label_dir ../VOC2012/pesudolabels_aug \
    --use_capsule \
    --capsule_loss_weight 0.1 \
    --primary_caps_num 32 \
    --primary_caps_dim 8 \
    --num_routing 3

echo "胶囊网络训练完成"

# 3. 小规模胶囊网络训练（适合资源有限的情况）
echo "3. 小规模胶囊网络训练..."
python train_voc_pseudo_capsule.py \
    --config ../configs/voc_attn_reg.yaml \
    --work_dir experiment_capsule_small \
    --crop_size 320 \
    --radius 8 \
    --pseudo_label_dir ../VOC2012/pesudolabels_aug \
    --use_capsule \
    --capsule_loss_weight 0.05 \
    --primary_caps_num 16 \
    --primary_caps_dim 4 \
    --num_routing 2

echo "小规模胶囊网络训练完成"

echo "=== 所有训练示例完成 ===" 