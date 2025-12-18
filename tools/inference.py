"""
Inference Script for FMCaps

Runs inference on images using trained FMCaps model.

Usage:
    python tools/inference.py \
        --checkpoint experiment_fmcaps_voc/checkpoints/best.pth \
        --input_dir images/ \
        --output_dir results/ \
        --config configs/voc_attn_reg.yaml
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf

# Import colormap encoding
from utils.imutils import encode_cmap


def load_model(checkpoint_path, config_path, device='cuda'):
    """
    Load trained FMCaps model
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        device: Device to load model on
        
    Returns:
        model: Loaded model
        cfg: Configuration
    """
    cfg = OmegaConf.load(config_path)
    
    # Determine model type based on config
    if 'voc' in config_path.lower():
        from WeCLIP_model.model_attn_aff_voc import WeCLIP
        num_classes = 21
    else:
        from WeCLIP_model.model_attn_aff_coco_capsule import WeCLIP
        num_classes = 81
    
    # Create model
    model = WeCLIP(
        num_classes=num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        device=device,
        use_capsule=True
    )
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    
    return model, cfg


def preprocess_image(image_path, crop_size=512):
    """
    Preprocess image for inference
    
    Args:
        image_path: Path to input image
        crop_size: Target size for the image
        
    Returns:
        tensor: Preprocessed image tensor
        original_size: Original image size (H, W)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size[::-1]  # (H, W)
    
    # Resize
    image = image.resize((crop_size, crop_size), Image.BILINEAR)
    
    # Convert to tensor and normalize
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std
    
    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
    
    return tensor, original_size


def inference_single(model, image_tensor, original_size, device='cuda'):
    """
    Run inference on a single image
    
    Args:
        model: FMCaps model
        image_tensor: Preprocessed image tensor
        original_size: Original image size (H, W)
        device: Device to run on
        
    Returns:
        prediction: Segmentation prediction (H, W)
    """
    with torch.no_grad():
        # Add batch dimension
        inputs = image_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        outputs = model(inputs, ['inference'], mode='val')
        
        # Get segmentation output
        if isinstance(outputs, tuple):
            segs = outputs[0]
        else:
            segs = outputs
        
        # Upsample to original size
        segs = F.interpolate(
            segs, 
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Get prediction
        prediction = torch.argmax(segs, dim=1).squeeze(0)
        prediction = prediction.cpu().numpy().astype(np.uint8)
    
    return prediction


def run_inference(
    checkpoint_path: str,
    config_path: str,
    input_dir: str,
    output_dir: str,
    save_colored: bool = True,
    device: str = 'cuda'
):
    """
    Run inference on a directory of images
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        input_dir: Directory containing input images
        output_dir: Directory to save predictions
        save_colored: Whether to save colored visualizations
        device: Device to run on
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_colored:
        colored_dir = os.path.join(output_dir, 'colored')
        os.makedirs(colored_dir, exist_ok=True)
    
    # Load model
    model, cfg = load_model(checkpoint_path, config_path, device)
    crop_size = cfg.dataset.crop_size if hasattr(cfg.dataset, 'crop_size') else 512
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        f for f in os.listdir(input_dir) 
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    for image_file in tqdm(image_files, desc="Running inference"):
        image_path = os.path.join(input_dir, image_file)
        
        # Preprocess
        image_tensor, original_size = preprocess_image(image_path, crop_size)
        
        # Inference
        prediction = inference_single(model, image_tensor, original_size, device)
        
        # Save prediction
        base_name = os.path.splitext(image_file)[0]
        pred_path = os.path.join(output_dir, f'{base_name}.png')
        Image.fromarray(prediction).save(pred_path)
        
        # Save colored visualization
        if save_colored:
            colored = encode_cmap(prediction)
            colored_path = os.path.join(colored_dir, f'{base_name}_colored.png')
            Image.fromarray(colored).save(colored_path)
    
    print(f"Predictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='FMCaps Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save predictions')
    parser.add_argument('--save_colored', action='store_true',
                        help='Save colored visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    
    args = parser.parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        save_colored=args.save_colored,
        device=args.device
    )


if __name__ == '__main__':
    main()

