#!/bin/bash
# FMCaps COCO Training Example Script

# Train with Capsule Network on MS COCO 2014
python coco_train_capsule_dic.py \
    --config ../configs/coco_attn_reg.yaml \
    --work_dir experiment_fmcaps_coco \
    --pseudo_label_dir ../MSCOCO/pseudo_labels \
    --crop_size 320 \
    --capsule_loss_weight 0.1

# Optional: Train without Capsule Network (baseline)
# python dist_clip_coco.py \
#     --config ../configs/coco_attn_reg.yaml \
#     --work_dir experiment_baseline_coco

