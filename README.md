# FMCaps: Integrating Foundation Models with Capsule Networks for Enhanced Weakly-Supervised Semantic Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the paper:

> **Integrating Foundation Models with Capsule Networks for Enhanced Weakly-Supervised Semantic Segmentation**
>
> *Expert Systems with Applications (ESWA)*

## 📋 Abstract

Weakly-Supervised Semantic Segmentation (WSSS) strives to achieve dense pixel-level predictions using only image-level annotations. This paper introduces **FMCaps**, a novel framework that synergistically integrates foundational models (CLIP, SAM, Grounding-DINO) with a capsule network module for enhanced performance.

**Key Contributions:**
- **SGFR Module**: SAM and Grounding-DINO Fusion Refinement for high-quality pseudo-label generation
- **SGAE Module**: SAM-Guided Affinity Enhancement for robust affinity learning
- **Capsule Network Integration**: Dynamic routing for structured object representations

**Results:**
| Dataset | Split | mIoU |
|---------|-------|------|
| PASCAL VOC 2012 | val | 78.2% |
| PASCAL VOC 2012 | test | 78.7% |
| MS COCO 2014 | val | 48.9% |

## 🏗️ Framework Overview

```
                                    FMCaps Framework
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   Image + Class Labels                                                  │
    │         │                                                               │
    │         ▼                                                               │
    │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
    │   │   WeCLIP    │────▶│    SGFR     │────▶│  Pseudo     │              │
    │   │  (CLIP+CAM) │     │ (SAM+GDINO) │     │  Labels     │              │
    │   └─────────────┘     └─────────────┘     └──────┬──────┘              │
    │         │                                        │                      │
    │         ▼                                        ▼                      │
    │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
    │   │  Encoder    │────▶│    SGAE     │────▶│  Affinity   │              │
    │   │  Features   │     │ (SAM Prior) │     │  Learning   │              │
    │   └──────┬──────┘     └─────────────┘     └─────────────┘              │
    │          │                                                              │
    │          ▼                                                              │
    │   ┌─────────────┐     ┌─────────────┐                                  │
    │   │  Capsule    │────▶│   Fusion    │────▶ Final Segmentation          │
    │   │  Network    │     │  Decoder    │                                  │
    │   └─────────────┘     └─────────────┘                                  │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
FMCaps/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── configs/                    # Configuration files
│   ├── voc_attn_reg.yaml      # PASCAL VOC config
│   └── coco_attn_reg.yaml     # MS COCO config
├── datasets/                   # Dataset loaders
│   ├── voc.py                 # PASCAL VOC dataset
│   ├── coco.py                # MS COCO dataset
│   └── transforms.py          # Data augmentation
├── WeCLIP_model/              # Core model components
│   ├── model_attn_aff_voc.py  # WeCLIP for VOC
│   ├── model_attn_aff_coco.py # WeCLIP for COCO
│   ├── capsule_module.py      # Capsule network module
│   ├── segformer_head.py      # Segmentation head
│   └── Decoder/               # Transformer decoder
├── modules/                    # FMCaps modules
│   ├── sgfr.py                # SGFR: SAM+Grounding-DINO Fusion
│   └── sgae.py                # SGAE: SAM-Guided Affinity
├── tools/                      # Utility scripts
│   ├── generate_pseudo_labels.py  # Pseudo-label generation
│   └── visualize_*.py         # Visualization tools
├── train_voc/                  # VOC training scripts
├── train_coco/                 # COCO training scripts
├── test_voc/                   # VOC evaluation scripts
├── test_coco/                  # COCO evaluation scripts
└── utils/                      # Utility functions
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/FMCaps.git
cd FMCaps

# Create conda environment
conda create -n fmcaps python=3.8
conda activate fmcaps

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install dependencies
pip install -r requirements.txt

# Install Grounding-DINO (optional, for SGFR)
pip install groundingdino-py

# Install SAM (optional, for SGFR and SGAE)
pip install segment-anything
```

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

## 🚀 Usage

### Step 1: Generate High-Quality Pseudo Labels (SGFR)

```bash
# For PASCAL VOC
python tools/generate_pseudo_labels.py \
    --dataset voc \
    --data_root ./VOC2012 \
    --output_dir ./VOC2012/pseudo_labels_sgfr \
    --use_sam --use_gdino

# For MS COCO
python tools/generate_pseudo_labels.py \
    --dataset coco \
    --data_root ./MSCOCO \
    --output_dir ./MSCOCO/pseudo_labels_sgfr \
    --use_sam --use_gdino
```

### Step 2: Train Segmentation Network

```bash
# Train on PASCAL VOC with Capsule Network
python train_voc/voc_train_capsule_dic.py \
    --config configs/voc_attn_reg.yaml \
    --work_dir experiment_fmcaps_voc \
    --pseudo_label_dir ./VOC2012/pseudo_labels_sgfr

# Train on MS COCO with Capsule Network
python train_coco/coco_train_capsule_dic.py \
    --config configs/coco_attn_reg.yaml \
    --work_dir experiment_fmcaps_coco \
    --pseudo_label_dir ./MSCOCO/pseudo_labels_sgfr
```

### Step 3: Evaluation

```bash
# Evaluate on PASCAL VOC val set
python test_voc/test_msc_flip_voc.py \
    --config configs/voc_attn_reg.yaml \
    --checkpoint experiment_fmcaps_voc/checkpoints/best.pth \
    --save_dir results/voc_val

# Evaluate on MS COCO val set
python test_coco/test_msc_flip_coco.py \
    --config configs/coco_attn_reg.yaml \
    --checkpoint experiment_fmcaps_coco/checkpoints/best.pth \
    --save_dir results/coco_val
```

## 🔬 Module Details

### SGFR: SAM and Grounding-DINO Fusion Refinement

The SGFR module generates high-quality pseudo-labels by:
1. Using **Grounding-DINO** for open-set object detection with class name prompts
2. Extracting **salient point prompts** from CAM local peaks
3. Prompting **SAM** with both boxes and points for precise segmentation
4. Applying **conflict resolution** to aggregate class-specific masks

```python
from modules.sgfr import SGFR

sgfr = SGFR(
    sam_checkpoint="sam_vit_h_4b8939.pth",
    gdino_checkpoint="groundingdino_swint_ogc.pth",
    device="cuda"
)

pseudo_labels = sgfr.generate(
    images=images,
    class_names=["cat", "dog", "person"],
    cams=initial_cams
)
```

### SGAE: SAM-Guided Affinity Enhancement

The SGAE module leverages SAM's structural priors to guide affinity learning:

```python
from modules.sgae import SGAE

sgae = SGAE(
    sam_checkpoint="sam_vit_h_4b8939.pth",
    device="cuda"
)

affinity_target = sgae.compute_affinity(
    images=images,
    features=encoder_features
)
```

### Capsule Network Module

Dynamic routing capsule network for structured object representations:

```python
from WeCLIP_model.capsule_module import create_capsule_module

capsule = create_capsule_module(
    in_channels=256,
    num_classes=21,
    primary_caps_num=32,
    primary_caps_dim=8,
    num_routing=3
)

outputs, enhanced_features = capsule(features, return_enhanced_features=True)
```

## 📊 Results

### PASCAL VOC 2012

| Method | Backbone | val mIoU | test mIoU |
|--------|----------|----------|-----------|
| WeCLIP (Baseline) | ViT-B/16 | 75.0 | 75.7 |
| **FMCaps (Ours)** | ViT-B/16 | **78.2** | **78.7** |

### MS COCO 2014

| Method | Backbone | val mIoU |
|--------|----------|----------|
| WeCLIP (Baseline) | ViT-B/16 | 45.3 |
| **FMCaps (Ours)** | ViT-B/16 | **48.9** |

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{fmcaps2025,
  title={Integrating Foundation Models with Capsule Networks for Enhanced Weakly-Supervised Semantic Segmentation},
  author={Guan, Wei and Yao, Zhuang and Li, Chengxin and Wu, Gengshen and Liu, Yi and Xu, Shoukun},
  journal={Expert Systems with Applications},
  year={2025},
  publisher={Elsevier}
}
```

## 🙏 Acknowledgements

This project builds upon the following excellent works:
- [CLIP](https://github.com/openai/CLIP) - OpenAI's Contrastive Language-Image Pre-training
- [WeCLIP](https://github.com/zbf1991/WeCLIP) - Baseline WSSS method with CLIP
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - Meta's foundation model for segmentation
- [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO) - Open-set object detection

## 📄 License

This project is released under the [MIT License](LICENSE).

## 📧 Contact

For questions or issues, please open a GitHub issue or contact:
- Wei Guan - [email]
- School of Computer Science and Artificial Intelligence, Changzhou University

