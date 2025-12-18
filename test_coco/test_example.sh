#!/bin/bash
# FMCaps COCO Testing Example Script

# Test on MS COCO 2014 val set
python test_msc_flip_coco.py \
    --config ../configs/coco_attn_reg.yaml \
    --checkpoint ../checkpoints/fmcaps_coco_best.pth \
    --save_dir ../results/coco_val

