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
# 解决中文字体显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用英文字体
plt.rcParams['axes.unicode_minus'] = False

from datasets import voc
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from utils.imutils import tensorboard_image, tensorboard_label, denormalize_img, encode_cmap
from WeCLIP_model.model_attn_aff_voc import WeCLIP

warnings.filterwarnings("ignore", category=UserWarning, message=".*MMCV will release v2.0.0.*")
# 忽略 PyTorch upsample 的警告 (如果确定是外部库引起的且不想修改)
warnings.filterwarnings("ignore", message="nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='../configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default='experiment_O', type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
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

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def save_val_visualization(inputs, segs, cam, labels, names, save_dir, iter_num, max_samples=5):
    """保存验证过程中的可视化图像"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    val_vis_dir = os.path.join(save_dir, 'validation')
    if not os.path.exists(val_vis_dir):
        os.makedirs(val_vis_dir, exist_ok=True)
    
    batch_size = min(max_samples, inputs.size(0))
    
    # 反归一化输入图像
    denorm_imgs = denormalize_img(inputs[:batch_size])
    
    # 预测结果
    pred_segs = torch.argmax(segs[:batch_size], dim=1)
    
    # CAM热力图
    cam_vis = cam[:batch_size]
    
    # 真实标签
    gt_labels = labels[:batch_size]
    
    for i in range(batch_size):
        try:
            # 创建子图
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # 原始图像
            img = denorm_imgs[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # CAM热力图 - 处理不同维度
            if cam_vis[i].dim() == 3:  # (num_classes, H, W)
                cam_np = cam_vis[i].max(dim=0)[0].cpu().numpy()
            elif cam_vis[i].dim() == 2:  # (H, W)
                cam_np = cam_vis[i].cpu().numpy()
            else:
                # 如果维度不匹配，跳过这个样本
                plt.close()
                continue
                
            # 归一化CAM到0-1范围
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            cam_colored = plt.cm.jet(cam_np)[:, :, :3]
            # 将CAM叠加到原图上
            cam_overlay = 0.6 * img/255.0 + 0.4 * cam_colored
            axes[1].imshow(cam_overlay)
            axes[1].set_title('CAM Heatmap')
            axes[1].axis('off')
            
            # 预测分割结果
            pred_seg = pred_segs[i].cpu().numpy()
            pred_colored = encode_cmap(pred_seg)
            axes[2].imshow(pred_colored)
            axes[2].set_title('Predicted Segmentation')
            axes[2].axis('off')
            
            # 真实标签
            gt_np = gt_labels[i].cpu().numpy()
            gt_colored = encode_cmap(gt_np)
            axes[3].imshow(gt_colored)
            axes[3].set_title('Ground Truth')
            axes[3].axis('off')
            
            # 保存图像
            img_name = names[i] if i < len(names) else f'val_sample_{i}'
            # 清理文件名，移除不安全字符
            img_name = str(img_name).replace('/', '_').replace('\\', '_').replace('.jpg', '').replace('.png', '')
            save_path = os.path.join(val_vis_dir, f'val_iter_{iter_num:06d}_{img_name}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            # 如果某个样本处理失败，记录错误但继续处理其他样本
            logging.warning(f"处理验证第{i}个样本时出错: {e}")
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

        segs, cam, attn_loss = model(inputs, name, 'val')

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

        preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
        cams += list(cam.cpu().numpy().astype(np.int16))
        gts += list(labels.cpu().numpy().astype(np.int16))
        
        # 保存验证可视化（只在前几个batch保存）
        if save_vis and num <= 3:
            try:
                save_val_visualization(inputs, resized_segs, cam, labels, name, cfg.work_dir.vis_dir, iter_num)
            except Exception as e:
                logging.warning(f"保存验证可视化失败: {e}")

        num+=1

        if num % 1000 ==0:
            seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
            cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
            preds, gts, cams, aff_gts = [], [], [], []

    seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
    cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
    model.train()
    return seg_score, cam_score

def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5

def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask

def save_visualization(inputs, segs, cam, pseudo_label, img_names, save_dir, iter_num, batch_size=4):
    """保存训练过程中的可视化图像"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # 限制batch大小，避免图像过多
    batch_size = min(batch_size, inputs.size(0))
    
    # 反归一化输入图像
    denorm_imgs = denormalize_img(inputs[:batch_size])
    
    # 预测结果
    pred_segs = torch.argmax(segs[:batch_size], dim=1)
    
    # CAM热力图 - 根据实际维度处理
    cam_vis = cam[:batch_size]
    
    # 伪标签 - 根据实际维度处理
    pseudo_vis = pseudo_label[:batch_size]
    
    for i in range(batch_size):
        try:
            # 创建子图
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # 原始图像
            img = denorm_imgs[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # CAM热力图 - 处理不同维度
            if cam_vis[i].dim() == 3:  # (num_classes, H, W)
                cam_np = cam_vis[i].max(dim=0)[0].cpu().numpy()
            elif cam_vis[i].dim() == 2:  # (H, W)
                cam_np = cam_vis[i].cpu().numpy()
            else:
                # 如果维度不匹配，跳过这个样本
                plt.close()
                continue
                
            # 归一化CAM到0-1范围
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            cam_colored = plt.cm.jet(cam_np)[:, :, :3]
            # 将CAM叠加到原图上
            cam_overlay = 0.6 * img/255.0 + 0.4 * cam_colored
            axes[1].imshow(cam_overlay)
            axes[1].set_title('CAM Heatmap')
            axes[1].axis('off')
            
            # 预测分割结果
            pred_seg = pred_segs[i].cpu().numpy()
            pred_colored = encode_cmap(pred_seg)
            axes[2].imshow(pred_colored)
            axes[2].set_title('Predicted Segmentation')
            axes[2].axis('off')
            
            # 伪标签
            pseudo_np = pseudo_vis[i].cpu().numpy()
            pseudo_colored = encode_cmap(pseudo_np)
            axes[3].imshow(pseudo_colored)
            axes[3].set_title('Pseudo Label')
            axes[3].axis('off')
            
            # 保存图像
            img_name = img_names[i] if i < len(img_names) else f'sample_{i}'
            # 清理文件名，移除不安全字符
            img_name = str(img_name).replace('/', '_').replace('\\', '_').replace('.jpg', '').replace('.png', '')
            save_path = os.path.join(save_dir, f'iter_{iter_num:06d}_{img_name}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            # 如果某个样本处理失败，记录错误但继续处理其他样本
            logging.warning(f"处理第{i}个样本时出错: {e}")
            if 'fig' in locals():
                plt.close(fig)
            continue
        
        # 只保存前几张图像，避免占用太多空间
        if i >= 2:  # 每次最多保存3张图像
            break

def train(cfg):

    num_workers = 10
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset = voc.VOC12ClsDataset(
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
        num_classes=cfg.dataset.num_classes,
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

    WeCLIP_model = WeCLIP(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        device='cuda'
    )
    ##########################################################################################################################################
    # logging.info('\nNetwork config: \n%s'%(WeCLIP_model))

    param_groups = WeCLIP_model.get_param_groups()
    WeCLIP_model.cuda()


    mask_size = int(cfg.dataset.crop_size // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)

    optimizer = PolyWarmupAdamW(
        params=[
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
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    #####################################################################################################
    # logging.info('\nOptimizer: \n%s' % optimizer)

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()


    for n_iter in range(cfg.train.max_iters):
        
        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        segs, cam, attn_pred = WeCLIP_model(inputs.cuda(), img_name)

        pseudo_label = cam

        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        fts_cam = cam.clone()

        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        seg_loss = get_seg_loss(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)

        loss = 1 * seg_loss + 0.1*attn_loss


        avg_meter.add({'seg_loss': seg_loss.item(), 'attn_loss': attn_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (n_iter + 1) % cfg.train.log_iters == 0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs,dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds==gts).sum()/preds.size

            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, attn_loss: %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('attn_loss'), seg_mAcc))

            writer.add_scalars('train/loss',  {"seg_loss": seg_loss.item(), "attn_loss": attn_loss.item()}, global_step=n_iter)
            
            # 保存可视化图像（每500次迭代保存一次）
            if (n_iter + 1) % (cfg.train.log_iters * 10) == 0:
                try:
                    # 添加调试信息
                    logging.info(f"可视化调试信息 - inputs: {inputs.shape}, segs: {segs.shape}, cam: {cam.shape}, pseudo_label: {pseudo_label.shape}")
                    save_visualization(inputs, segs, cam, pseudo_label, img_name, cfg.work_dir.vis_dir, n_iter+1)
                    logging.info(f"保存可视化图像到 {cfg.work_dir.vis_dir}")
                except Exception as e:
                    logging.warning(f"保存可视化图像失败: {e}")
                    import traceback
                    logging.warning(f"详细错误信息: {traceback.format_exc()}")

        
        if (n_iter + 1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "WeCLIP_model_iter_%d.pth"%(n_iter+1))
            logging.info('Validating...')
            if (n_iter + 1) > 26000:
                torch.save(WeCLIP_model.state_dict(), ckpt_name)
            # 每隔几次验证保存一次可视化
            save_vis = (n_iter + 1) % (cfg.train.eval_iters * 2) == 0
            seg_score, cam_score = validate(model=WeCLIP_model, data_loader=val_loader, cfg=cfg, 
                                          save_vis=save_vis, iter_num=n_iter+1)
            logging.info("cams score:")
            logging.info(cam_score)
            logging.info("segs score:")
            logging.info(seg_score)

    return True

if __name__ == "__main__":

    args = parser.parse_args()
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

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    train(cfg=cfg)
