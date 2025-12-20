import argparse
import datetime
import logging
import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys

sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# Font settings for visualization
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from datasets import voc
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from utils.imutils import tensorboard_image, tensorboard_label, denormalize_img, encode_cmap

# Import WeCLIP model
from WeCLIP_model.model_attn_aff_voc import WeCLIP

warnings.filterwarnings("ignore", category=UserWarning, message=".*MMCV will release v2.0.0.*")
warnings.filterwarnings("ignore",
                        message="nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='../configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default='experiment_fmcaps_voc', type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")
parser.add_argument("--pseudo_label_dir", default="../VOC2012/pesudolabels_aug", type=str,
                    help="pseudo label directory")
parser.add_argument("--capsule_loss_weight", default=0.1, type=float, help="capsule loss weight")
parser.add_argument("--use_capsule", type=bool, default=True, help="enable capsule network")
parser.add_argument("--disable_capsule", action="store_true", help="disable capsule network (overrides use_capsule)")
parser.add_argument("--primary_caps_num", default=32, type=int, help="number of primary capsules")
parser.add_argument("--primary_caps_dim", default=8, type=int, help="dimension of primary capsules")
parser.add_argument("--num_routing", default=3, type=int, help="number of routing iterations")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(filename='test.log'):
    """Setup logger"""
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def save_val_visualization(inputs, segs, cam, labels, names, save_dir, iter_num, max_samples=5):
    """Save visualization images during validation"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    val_vis_dir = os.path.join(save_dir, 'validation')
    if not os.path.exists(val_vis_dir):
        os.makedirs(val_vis_dir, exist_ok=True)

    batch_size = min(max_samples, inputs.size(0))

    # Denormalize input images
    denorm_imgs = denormalize_img(inputs[:batch_size])

    # Prediction results
    pred_segs = torch.argmax(segs[:batch_size], dim=1)

    # CAM heatmap
    cam_vis = cam[:batch_size]

    # Ground truth labels
    gt_labels = labels[:batch_size]

    for i in range(batch_size):
        try:
            # Create subplots
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # Original image
            img = denorm_imgs[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # CAM heatmap - handle different dimensions
            if cam_vis[i].dim() == 3:  # (num_classes, H, W)
                cam_np = cam_vis[i].max(dim=0)[0].detach().cpu().numpy()
            elif cam_vis[i].dim() == 2:  # (H, W)
                cam_np = cam_vis[i].detach().cpu().numpy()
            else:
                # Skip this sample if dimensions don't match
                plt.close()
                continue

            # Normalize CAM to 0-1 range
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            cam_colored = plt.cm.jet(cam_np)[:, :, :3]
            # Overlay CAM on original image
            cam_overlay = 0.6 * img / 255.0 + 0.4 * cam_colored
            axes[1].imshow(cam_overlay)
            axes[1].set_title('CAM Heatmap')
            axes[1].axis('off')

            # Predicted segmentation
            pred_seg = pred_segs[i].detach().cpu().numpy()
            pred_colored = encode_cmap(pred_seg)
            axes[2].imshow(pred_colored)
            axes[2].set_title('Predicted Segmentation')
            axes[2].axis('off')

            # Ground truth
            gt_np = gt_labels[i].detach().cpu().numpy()
            gt_colored = encode_cmap(gt_np)
            axes[3].imshow(gt_colored)
            axes[3].set_title('Ground Truth')
            axes[3].axis('off')

            # Save image
            img_name = names[i] if i < len(names) else f'val_sample_{i}'
            # Clean filename, remove unsafe characters
            img_name = str(img_name).replace('/', '_').replace('\\', '_').replace('.jpg', '').replace('.png', '')
            save_path = os.path.join(val_vis_dir, f'val_iter_{iter_num:06d}_{img_name}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            # Log error but continue processing other samples
            logging.warning(f"Error processing validation sample {i}: {e}")
            if 'fig' in locals():
                plt.close(fig)
            continue


def validate(model=None, data_loader=None, cfg=None, save_vis=False, iter_num=0):
    preds, gts, cams, aff_gts = [], [], [], []
    num = 1
    seg_hist = np.zeros((21, 21))
    cam_hist = np.zeros((21, 21))
    for _, data in tqdm(enumerate(data_loader),
                        total=len(data_loader), ncols=100, ascii=" >="):
        name, inputs, labels, cls_label = data

        inputs = inputs.cuda()
        labels = labels.cuda()

        # Validation mode, no capsule network output
        segs, cam, attn_loss = model(inputs, name, 'val')

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

        preds += list(torch.argmax(resized_segs, dim=1).detach().cpu().numpy().astype(np.int16))
        cams += list(cam.detach().cpu().numpy().astype(np.int16))
        gts += list(labels.detach().cpu().numpy().astype(np.int16))

        # Save validation visualization (only for first few batches)
        if save_vis and num <= 3:
            try:
                save_val_visualization(inputs, resized_segs, cam, labels, name, cfg.work_dir.vis_dir, iter_num)
            except Exception as e:
                logging.warning(f"Failed to save validation visualization: {e}")

        num += 1

        if num % 1000 == 0:
            seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
            cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
            preds, gts, cams, aff_gts = [], [], [], []

    seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
    cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
    model.train()
    return seg_score, cam_score


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius + 1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius + 1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask


def save_visualization(inputs, segs, cam, pseudo_label, img_names, save_dir, iter_num, batch_size=4,
                       capsule_outputs=None):
    """Save visualization images during training, including capsule network output"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Limit batch size to avoid too many images
    batch_size = min(batch_size, inputs.size(0))

    # Denormalize input images
    denorm_imgs = denormalize_img(inputs[:batch_size])

    # Prediction results
    pred_segs = torch.argmax(segs[:batch_size], dim=1)

    # CAM heatmap - handle actual dimensions
    cam_vis = cam[:batch_size]

    # Pseudo labels - handle actual dimensions
    pseudo_vis = pseudo_label[:batch_size]

    for i in range(batch_size):
        try:
            # Determine number of subplots based on capsule network output
            if capsule_outputs is not None and 'seg_logits' in capsule_outputs:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
            else:
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # Original image
            img = denorm_imgs[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # CAM heatmap - handle different dimensions
            if cam_vis[i].dim() == 3:  # (num_classes, H, W)
                cam_np = cam_vis[i].max(dim=0)[0].detach().cpu().numpy()
            elif cam_vis[i].dim() == 2:  # (H, W)
                cam_np = cam_vis[i].detach().cpu().numpy()
            else:
                # Skip this sample if dimensions don't match
                plt.close()
                continue

            # Normalize CAM to 0-1 range
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            cam_colored = plt.cm.jet(cam_np)[:, :, :3]
            # Overlay CAM on original image
            cam_overlay = 0.6 * img / 255.0 + 0.4 * cam_colored
            axes[1].imshow(cam_overlay)
            axes[1].set_title('CAM Heatmap')
            axes[1].axis('off')

            # Predicted segmentation
            pred_seg = pred_segs[i].detach().cpu().numpy()
            pred_colored = encode_cmap(pred_seg)
            axes[2].imshow(pred_colored)
            axes[2].set_title('Final Segmentation')
            axes[2].axis('off')

            # Pseudo label
            pseudo_np = pseudo_vis[i].detach().cpu().numpy()
            pseudo_colored = encode_cmap(pseudo_np)
            axes[3].imshow(pseudo_colored)
            axes[3].set_title('Pseudo Label')
            axes[3].axis('off')

            # Add extra visualization if capsule network output is available
            if capsule_outputs is not None and 'seg_logits' in capsule_outputs:
                # Capsule segmentation result
                capsule_seg_logits = capsule_outputs['seg_logits']
                if capsule_seg_logits.shape[-2:] != pred_segs.shape[-2:]:
                    capsule_seg_logits = F.interpolate(capsule_seg_logits, size=pred_segs.shape[-2:],
                                                       mode='bilinear', align_corners=False)
                capsule_seg = torch.argmax(capsule_seg_logits[i:i + 1], dim=1)[0].detach().cpu().numpy()
                capsule_colored = encode_cmap(capsule_seg)
                axes[4].imshow(capsule_colored)
                axes[4].set_title('Capsule Segmentation')
                axes[4].axis('off')

                # Capsule feature visualization (if primary capsule output available)
                if 'primary_caps' in capsule_outputs:
                    primary_caps = capsule_outputs['primary_caps'][i]  # [num_caps, H, W, dim]
                    # Compute capsule length as feature strength
                    caps_length = torch.sqrt((primary_caps ** 2).sum(dim=-1))  # [num_caps, H, W]
                    caps_vis = caps_length.mean(dim=0).detach().cpu().numpy()  # [H, W]
                    caps_vis = (caps_vis - caps_vis.min()) / (caps_vis.max() - caps_vis.min() + 1e-8)
                    axes[5].imshow(caps_vis, cmap='viridis')
                    axes[5].set_title('Capsule Features')
                    axes[5].axis('off')
                else:
                    axes[5].axis('off')

            # Save image
            img_name = img_names[i] if i < len(img_names) else f'sample_{i}'
            # Clean filename, remove unsafe characters
            img_name = str(img_name).replace('/', '_').replace('\\', '_').replace('.jpg', '').replace('.png', '')
            save_path = os.path.join(save_dir, f'iter_{iter_num:06d}_{img_name}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            # Log error but continue processing other samples
            logging.warning(f"Error processing sample {i}: {e}")
            if 'fig' in locals():
                plt.close(fig)
            continue

        # Only save first few images to avoid taking too much space
        if i >= 2:  # Save at most 3 images each time
            break


def train(cfg):
    num_workers = 10

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        pseudo_label_dir=args.pseudo_label_dir,  # Pseudo label directory
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='train',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    # Create WeCLIP model with capsule network integration
    capsule_config = None
    if args.use_capsule:
        capsule_config = {
            'feature_size': cfg.dataset.crop_size // 16,  # Calculate feature map size based on crop_size
            'primary_caps_num': args.primary_caps_num,
            'primary_caps_dim': args.primary_caps_dim,
            'class_caps_dim': 16,
            'num_routing': args.num_routing,
            'enable_segmentation': True,
            'enable_feature_enhancement': True
        }

    WeCLIP_model = WeCLIP(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        device='cuda',
        use_capsule=args.use_capsule,
        capsule_config=capsule_config
    )

    logging.info(f"WeCLIP model created, capsule network: {'enabled' if args.use_capsule else 'disabled'}")

    param_groups = WeCLIP_model.get_param_groups()
    WeCLIP_model.cuda()

    mask_size = int(cfg.dataset.crop_size // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)

    # Set different learning rates for capsule network parameters
    optimizer_params = [
        {
            "params": param_groups[0],
            "lr": cfg.optimizer.learning_rate,
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": param_groups[1],
            "lr": 0.0,
            "weight_decay": 0.0,
        },
        {
            "params": param_groups[2],
            "lr": cfg.optimizer.learning_rate * 10,
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": param_groups[3],
            "lr": cfg.optimizer.learning_rate * 10,
            "weight_decay": cfg.optimizer.weight_decay,
        },
    ]

    # Add capsule network parameter group if enabled
    if args.use_capsule and len(param_groups) > 4:
        optimizer_params.append({
            "params": param_groups[4],  # Capsule network parameters
            "lr": cfg.optimizer.learning_rate * 5,  # Higher learning rate for capsule network
            "weight_decay": cfg.optimizer.weight_decay,
        })

    optimizer = PolyWarmupAdamW(
        params=optimizer_params,
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    logging.info("Starting training loop")

    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs, pseudo_labels, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, pseudo_labels, cls_labels, img_box = next(train_loader_iter)

        # Forward pass - different outputs based on capsule network status
        outputs = WeCLIP_model(inputs.cuda(), img_name, mode='train')

        if args.use_capsule and len(outputs) == 4:
            # Capsule network enabled outputs
            segs, cam, attn_pred, capsule_outputs = outputs
        else:
            # Standard outputs
            segs, cam, attn_pred = outputs
            capsule_outputs = None

        # Use pseudo labels
        pseudo_labels = pseudo_labels.cuda()

        # Align prediction with pseudo label size
        segs = F.interpolate(segs, size=pseudo_labels.shape[1:], mode='bilinear', align_corners=False)

        # 1. Compute segmentation loss
        seg_loss = get_seg_loss(segs, pseudo_labels.type(torch.long), ignore_index=cfg.dataset.ignore_index)

        # 2. Compute capsule network loss (if enabled)
        capsule_loss = torch.tensor(0.0, device=segs.device)
        capsule_loss_dict = {}
        if args.use_capsule and capsule_outputs is not None:
            capsule_loss, capsule_loss_dict = WeCLIP_model.compute_capsule_loss(capsule_outputs, pseudo_labels)

        # Use original affinity loss
        fts = cam.clone()

        # 3. Compute affinity loss
        aff_label = cams_to_affinity_label(
            pseudo_labels.detach(),
            mask=attn_mask,
            ignore_index=cfg.dataset.ignore_index,
            clip_flag=16
        )
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        # 4. Compute total loss
        total_loss = (1.0 * seg_loss + args.capsule_loss_weight * capsule_loss + 0.1 * attn_loss)

        avg_meter.add({
            'seg_loss': seg_loss.item(),
            'attn_loss': attn_loss.item(),
            'capsule_loss': capsule_loss.item(),
            'total_loss': total_loss.item()
        })

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (n_iter + 1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs, dim=1).detach().cpu().numpy().astype(np.int16)
            gts = pseudo_labels.detach().cpu().numpy().astype(np.int16)

            seg_mAcc = (preds == gts).sum() / preds.size

            logging.info(
                "Iter: %d; Elapsed: %s; ETA: %s; LR: %.3e; seg_loss: %.4f, attn_loss: %.4f, capsule_loss: %.4f, total_loss: %.4f, seg_mAcc: %.4f" % (
                    n_iter + 1, delta, eta, cur_lr,
                    avg_meter.pop('seg_loss'),
                    avg_meter.pop('attn_loss'),
                    avg_meter.pop('capsule_loss'),
                    avg_meter.pop('total_loss'),
                    seg_mAcc))

            writer.add_scalars('train/loss', {
                "seg_loss": seg_loss.item(),
                "attn_loss": attn_loss.item(),
                "capsule_loss": capsule_loss.item(),
                "total_loss": total_loss.item()
            }, global_step=n_iter)

            # Log capsule network loss details to tensorboard if available
            if capsule_loss_dict:
                writer.add_scalars('train/capsule_loss_detail', capsule_loss_dict, global_step=n_iter)

            # Save visualization images (every 500 iterations)
            if (n_iter + 1) % (cfg.train.log_iters * 10) == 0:
                try:
                    logging.info(
                        f"Visualization debug info - inputs: {inputs.shape}, segs: {segs.shape}, cam: {cam.shape}, pseudo_label: {pseudo_labels.shape}")
                    save_visualization(inputs, segs, cam, pseudo_labels, img_name, cfg.work_dir.vis_dir,
                                       n_iter + 1, capsule_outputs=capsule_outputs)
                    logging.info(f"Saved visualization images to {cfg.work_dir.vis_dir}")
                except Exception as e:
                    logging.warning(f"Failed to save visualization images: {e}")
                    import traceback
                    logging.warning(f"Detailed error: {traceback.format_exc()}")

        if (n_iter + 1) % cfg.train.eval_iters == 0:
            model_name = "WeCLIP_capsule_model" if args.use_capsule else "WeCLIP_model"
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, f"{model_name}_iter_{n_iter + 1}.pth")
            logging.info('Validating...')
            if (n_iter + 1) > 28000:
                torch.save(WeCLIP_model.state_dict(), ckpt_name)
                logging.info(f"Model saved to: {ckpt_name}")

            # Save visualization every few validations
            save_vis = (n_iter + 1) % (cfg.train.eval_iters * 2) == 0
            seg_score, cam_score = validate(model=WeCLIP_model, data_loader=val_loader, cfg=cfg,
                                            save_vis=save_vis, iter_num=n_iter + 1)
            logging.info("cams score:")
            logging.info(cam_score)
            logging.info("segs score:")
            logging.info(seg_score)

    logging.info("Training completed")
    return True


if __name__ == "__main__":

    args = parser.parse_args()

    # Handle capsule network parameter logic
    if args.disable_capsule:
        args.use_capsule = False

    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)
    cfg.work_dir.vis_dir = os.path.join(cfg.work_dir.dir, 'visualizations', timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.vis_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp + '.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)
    logging.info(
        f'\nCapsule network params: capsule_loss_weight={args.capsule_loss_weight}, use_capsule={args.use_capsule}, primary_caps_num={args.primary_caps_num}, primary_caps_dim={args.primary_caps_dim}, num_routing={args.num_routing}')

    setup_seed(1)
    train(cfg=cfg)
