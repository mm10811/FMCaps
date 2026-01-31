# FMCaps: Integrating Foundation Models with Capsule Networks for Enhanced Weakly-Supervised Semantic Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 📋 Abstract

Weakly-Supervised Semantic Segmentation (WSSS) strives to achieve dense pixel-level predictions using only image-level annotations. This paper introduces **FMCaps**, a novel framework that synergistically integrates foundational models (CLIP, SAM, Grounding-DINO) with a capsule network module for enhanced performance.

**Key Contributions:**
- **SGFR Module**: SAM and Grounding-DINO Fusion Refinement for high-quality pseudo-label generation
- **SGAE Module**: SAM-Guided Affinity Enhancement for robust affinity learning
- **Capsule Network Integration**: Dynamic routing for structured object representations

**Results:**
| Dataset | Split | mIoU |
|---------|-------|------|
| PASCAL VOC 2012 | val | 79.0% |
| PASCAL VOC 2012 | test | 78.7% |
| MS COCO 2014 | val |57.5% |

## 🏗️ Framework Overview
<p align="center">
  <img src="./fig/overview.png" width="800" alt="FMCaps Framework Architecture"/>
</p>

## 🎨 Pseudo-Label Visualization

### PASCAL VOC 2012
<p align="center">
  <img src="./fig/pseudo_label_voc.png" width="800" alt="Pseudo-label visualization on PASCAL VOC 2012"/>
</p>

### MS COCO 2014
<p align="center">
  <img src="./fig/pseudo_label_coco.png" width="800" alt="Pseudo-label visualization on MS COCO 2014"/>
</p>

## 🔧 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+

## 📦 Data Preparation

### PASCAL VOC 2012

```bash
# Download VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar

# Download augmented annotations (SBD)
# Place in VOC2012/SegmentationClassAug/

# Expected structure:
VOC2012/
├── JPEGImages/
├── SegmentationClass/
├── SegmentationClassAug/
└── ImageSets/
    └── Segmentation/
        ├── train.txt
        ├── trainval.txt
        └── val.txt
```

### MS COCO 2014

```bash
# Download COCO 2014
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

# Expected structure:
MSCOCO/
├── JPEGImages/
│   ├── train/
│   └── val/
├── SegmentationClass/
│   ├── train/
│   └── val/
└── ImageSets/
```

### Pre-trained Models

Download the CLIP pre-trained model:
```bash
mkdir -p pretrained
# Download ViT-B/16 from OpenAI CLIP
wget -P pretrained/ https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```
### Checkpoints of our model
voc:https://drive.google.com/file/d/1pnCglcXecQmxT2xKl0CDfBNZOVl_Fc1m/view?usp=drive_link
coco:https://drive.google.com/file/d/15dfMgnt4Ts60-_Whqi8RgMIkV49avIz7/view?usp=drive_link

## 🚀 Usage

### Train Segmentation Network

```bash
# Train on PASCAL VOC with Capsule Network
python train/voc_train_capsule.py 

# Train on MS COCO with Capsule Network
python train/coco_train_capsule.py
```

## 🙏 Acknowledgements

This project builds upon the following excellent works:
- [CLIP](https://github.com/openai/CLIP) - OpenAI's Contrastive Language-Image Pre-training
- [WeCLIP](https://github.com/zbf1991/WeCLIP) - Baseline WSSS method with CLIP
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - Meta's foundation model for segmentation
- [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO) - Open-set object detection

## 📄 License

This project is released under the [MIT License](LICENSE).

