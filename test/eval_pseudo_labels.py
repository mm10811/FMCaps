import argparse
import os
import sys
sys.path.append("..")
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import multiprocessing
from tqdm import tqdm
import joblib
from datasets import voc
from utils import evaluate
from WeCLIP_model.model_attn_aff_voc import WeCLIP
import imageio.v2 as imageio
import logging
import datetime

def setup_logger(log_file):
    """设置日志记录器"""
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 文件处理器
    fHandler = logging.FileHandler(log_file, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    # 控制台处理器
    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='../configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--work_dir", default="pseudo_eval_results", type=str, help="work_dir")
parser.add_argument("--resize_long", default=512, type=int, help="resize the long side")
parser.add_argument("--eval_set", default="train_aug", type=str, help="eval_set") # train_aug for pseudo labels
parser.add_argument("--model_path", default="../scripts/experiment_capsule/checkpoints/2025-05-30-16-12/WeCLIP_capsule_model_iter_58000.pth", type=str, help="model_path")
parser.add_argument("--save_pseudo_labels", action="store_true", help="Save generated pseudo labels")
parser.add_argument("--use_crf", action="store_true", help="Apply CRF post-processing")
parser.add_argument("--eval_type", default="cam", choices=["cam", "seg", "both"], help="What to evaluate: cam, seg, or both")

# Capsule Network Arguments
parser.add_argument("--use_capsule", type=bool, default=True, help="Enable capsule network (default: False for testing, override if model was trained with capsules)")
parser.add_argument("--disable_capsule", action="store_true", help="Explicitly disable capsule network (overrides use_capsule if both are set)")
parser.add_argument("--primary_caps_num", default=32, type=int, help="Number of primary capsules")
parser.add_argument("--primary_caps_dim", default=8, type=int, help="Dimension of primary capsules")
parser.add_argument("--num_routing", default=3, type=int, help="Number of routing iterations")

def generate_pseudo_labels(model, dataset, eval_type="cam"):
    """
    生成伪标签并计算mIoU
    """
    cam_preds, seg_preds, gts = [], [], []
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    model.cuda(0)
    model.eval()
    
    logging.info(f"开始生成伪标签，数据集大小: {len(dataset)}, 评估类型: {eval_type}")

    num = 0
    cam_hist = np.zeros((21, 21))
    seg_hist = np.zeros((21, 21))

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
        num += 1

        name, inputs, labels, cls_labels = data
        names = name + name

        inputs = inputs.cuda()
        labels = labels.cuda()

        # resize long side to specified size
        _, _, h, w = inputs.shape
        ratio = args.resize_long / max(h, w)
        _h, _w = int(h * ratio), int(w * ratio)
        inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)

        # Forward pass
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        segs_cat, cam, attn_loss = model(inputs_cat, names, mode='val')
        
        # Process CAM
        cam = cam[0].unsqueeze(0)
        
        # Process segmentation
        segs = segs_cat[0].unsqueeze(0)
        _segs = (segs_cat[0, ...] + segs_cat[1, ...].flip(-1)) / 2
        final_segs = _segs.unsqueeze(0)

        # Resize to original label size
        resized_cam = F.interpolate(cam, size=labels.shape[1:], mode='bilinear', align_corners=False)
        cam_pred = torch.argmax(resized_cam, dim=1)

        resized_segs = F.interpolate(final_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_pred = torch.argmax(resized_segs, dim=1)

        # Save pseudo labels if requested
        if args.save_pseudo_labels:
            if eval_type in ["cam", "both"]:
                cam_save_path = os.path.join(args.work_dir, "pseudo_labels_cam", name[0] + ".png")
                imageio.imsave(cam_save_path, np.squeeze(cam_pred.cpu().numpy()).astype(np.uint8))
            
            if eval_type in ["seg", "both"]:
                seg_save_path = os.path.join(args.work_dir, "pseudo_labels_seg", name[0] + ".png")
                imageio.imsave(seg_save_path, np.squeeze(seg_pred.cpu().numpy()).astype(np.uint8))

        # Collect predictions
        if eval_type in ["cam", "both"]:
            cam_preds += list(cam_pred.cpu().numpy().astype(np.int16))
        
        if eval_type in ["seg", "both"]:
            seg_preds += list(seg_pred.cpu().numpy().astype(np.int16))
        
        gts += list(labels.cpu().numpy().astype(np.int16))

        # Periodic evaluation
        if num % 500 == 0:
            current_gts = gts[-500:] if num >= 500 else gts
            
            if eval_type in ["cam", "both"] and len(cam_preds) > 0:
                current_cam_preds = cam_preds[-500:] if num >= 500 else cam_preds
                cam_hist, cam_score = evaluate.scores(current_gts, current_cam_preds, cam_hist)
                logging.info(f"已处理 {num} 张图像 - CAM mIoU: {cam_score['miou']:.4f}")
            
            if eval_type in ["seg", "both"] and len(seg_preds) > 0:
                current_seg_preds = seg_preds[-500:] if num >= 500 else seg_preds
                seg_hist, seg_score = evaluate.scores(current_gts, current_seg_preds, seg_hist)
                logging.info(f"已处理 {num} 张图像 - Seg mIoU: {seg_score['miou']:.4f}")

    logging.info("伪标签生成完成")
    return cam_preds, seg_preds, gts

def apply_crf_postprocessing(config, eval_type="cam"):
    """
    应用CRF后处理
    """
    print("CRF后处理...")
    logging.info("开始CRF后处理...")

    txt_name = os.path.join(config.dataset.name_list_dir, args.eval_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    logging.info(f"CRF处理图像数量: {len(name_list)}")

    images_path = os.path.join(config.dataset.root_dir, 'JPEGImages')
    labels_path = os.path.join(config.dataset.root_dir, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,
        pos_xy_std=3,
        pos_w=3,
        bi_xy_std=64,
        bi_rgb_std=5,
        bi_w=4,
    )

    def _job(i):
        name = name_list[i]
        
        # 根据评估类型选择不同的伪标签源
        if eval_type == "cam":
            pseudo_label_path = os.path.join(args.work_dir, "pseudo_labels_cam", name + ".png")
        else:  # seg
            pseudo_label_path = os.path.join(args.work_dir, "pseudo_labels_seg", name + ".png")
        
        if not os.path.exists(pseudo_label_path):
            return None, None
            
        pseudo_label = imageio.imread(pseudo_label_path)
        
        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        
        label_name = os.path.join(labels_path, name + ".png")
        if os.path.exists(label_name):
            label = imageio.imread(label_name)
        else:
            return None, None

        H, W, _ = image.shape
        
        # 将伪标签转换为概率分布以便CRF处理
        pseudo_prob = np.zeros((21, H, W))
        for c in range(21):
            pseudo_prob[c] = (pseudo_label == c).astype(np.float32)
        pseudo_prob = pseudo_prob + 1e-8  # 避免零概率
        pseudo_prob = pseudo_prob / pseudo_prob.sum(axis=0, keepdims=True)

        image = image.astype(np.uint8)
        refined_prob = post_processor(image, pseudo_prob)
        refined_pred = np.argmax(refined_prob, axis=0)

        # 保存CRF处理后的结果
        crf_output_dir = os.path.join(args.work_dir, f"pseudo_labels_{eval_type}_crf")
        os.makedirs(crf_output_dir, exist_ok=True)
        imageio.imsave(os.path.join(crf_output_dir, name + ".png"), refined_pred.astype(np.uint8))
        
        return refined_pred, label

    results = joblib.Parallel(n_jobs=4, verbose=10, pre_dispatch="all")(
        [joblib.delayed(_job)(i) for i in range(len(name_list))])

    # 过滤None结果
    valid_results = [(pred, gt) for pred, gt in results if pred is not None and gt is not None]
    if not valid_results:
        logging.warning("没有有效的CRF处理结果")
        return None
    
    preds, gts = zip(*valid_results)
    hist = np.zeros((21, 21))
    hist, score = evaluate.scores(gts, preds, hist, 21)

    logging.info("CRF后处理完成")
    logging.info(f"CRF后处理结果: {score}")
    print(f"CRF {eval_type} Score:", score)
    
    return score

def main(cfg):
    logging.info("开始伪标签评估流程")
    logging.info(f"配置信息: {cfg}")
    logging.info(f"参数信息: {args}")
    
    # 创建数据集
    dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    logging.info(f"数据集加载完成，数据集大小: {len(dataset)}")

    # 处理胶囊网络参数逻辑
    actual_use_capsule = args.use_capsule
    if args.disable_capsule:
        actual_use_capsule = False

    capsule_config = None
    if actual_use_capsule:
        capsule_config = {
            'feature_size': cfg.dataset.crop_size // 16,
            'primary_caps_num': args.primary_caps_num,
            'primary_caps_dim': args.primary_caps_dim,
            'class_caps_dim': 16,
            'num_routing': args.num_routing,
            'enable_segmentation': True,
            'enable_feature_enhancement': True
        }
        logging.info(f"启用胶囊网络进行评估，配置: {capsule_config}")
    else:
        logging.info("未启用胶囊网络进行评估")

    # 创建模型
    WeCLIP_model = WeCLIP(num_classes=cfg.dataset.num_classes,
                         clip_model=cfg.clip_init.clip_pretrain_path,
                         embedding_dim=cfg.clip_init.embedding_dim,
                         in_channels=cfg.clip_init.in_channels,
                         dataset_root_path=cfg.dataset.root_dir,
                         device='cuda',
                         use_capsule=actual_use_capsule,
                         capsule_config=capsule_config)
    
    logging.info(f"模型创建完成，开始加载权重: {args.model_path}")
    
    trained_state_dict = torch.load(args.model_path, map_location="cpu")
    WeCLIP_model.load_state_dict(state_dict=trained_state_dict, strict=False)
    WeCLIP_model.eval()
    
    logging.info("模型权重加载完成")

    # 生成伪标签
    cam_preds, seg_preds, gts = generate_pseudo_labels(model=WeCLIP_model, dataset=dataset, eval_type=args.eval_type)
    torch.cuda.empty_cache()

    # 计算最终评估指标
    results = {}
    
    if args.eval_type in ["cam", "both"] and len(cam_preds) > 0:
        cam_hist = np.zeros((21, 21))
        cam_hist, cam_score = evaluate.scores(gts, cam_preds, cam_hist)
        results['cam'] = cam_score
        logging.info(f"CAM伪标签最终结果: {cam_score}")
        print("CAM Pseudo Labels Score:")
        print(cam_score)
    
    if args.eval_type in ["seg", "both"] and len(seg_preds) > 0:
        seg_hist = np.zeros((21, 21))
        seg_hist, seg_score = evaluate.scores(gts, seg_preds, seg_hist)
        results['seg'] = seg_score
        logging.info(f"Seg伪标签最终结果: {seg_score}")
        print("Seg Pseudo Labels Score:")
        print(seg_score)

    # 应用CRF后处理（如果启用）
    if args.use_crf and args.save_pseudo_labels:
        if args.eval_type in ["cam", "both"]:
            crf_cam_score = apply_crf_postprocessing(config=cfg, eval_type="cam")
            if crf_cam_score:
                results['cam_crf'] = crf_cam_score
        
        if args.eval_type in ["seg", "both"]:
            crf_seg_score = apply_crf_postprocessing(config=cfg, eval_type="seg")
            if crf_seg_score:
                results['seg_crf'] = crf_seg_score

    # 保存评估结果
    results_file = os.path.join(args.work_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("=== 伪标签评估结果 ===\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"评估集: {args.eval_set}\n")
        f.write(f"评估类型: {args.eval_type}\n")
        f.write(f"使用CRF: {args.use_crf}\n")
        f.write(f"胶囊网络: {actual_use_capsule}\n")
        f.write("\n")
        
        for eval_name, score in results.items():
            f.write(f"{eval_name.upper()} Results:\n")
            f.write(f"  mIoU: {score['miou']:.4f}\n")
            f.write(f"  Pixel Acc: {score['pacc']:.4f}\n")
            f.write(f"  Mean Acc: {score['macc']:.4f}\n")
            f.write("\n")

    logging.info("伪标签评估流程完成")
    logging.info(f"结果已保存到: {results_file}")
    return results

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    print("配置信息:")
    print(cfg)
    print("\n参数信息:")
    print(args)

    # 创建工作目录
    os.makedirs(args.work_dir, exist_ok=True)
    
    if args.save_pseudo_labels:
        if args.eval_type in ["cam", "both"]:
            os.makedirs(os.path.join(args.work_dir, "pseudo_labels_cam"), exist_ok=True)
        if args.eval_type in ["seg", "both"]:
            os.makedirs(os.path.join(args.work_dir, "pseudo_labels_seg"), exist_ok=True)
    
    # 设置日志文件
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    log_file = os.path.join(args.work_dir, f"pseudo_eval_log_{timestamp}.log")
    setup_logger(log_file)
    
    logging.info("="*50)
    logging.info("开始伪标签评估程序")
    logging.info(f"时间戳: {timestamp}")
    logging.info(f"模型路径: {args.model_path}")
    logging.info(f"评估集: {args.eval_set}")
    logging.info(f"评估类型: {args.eval_type}")
    logging.info(f"工作目录: {args.work_dir}")
    logging.info(f"日志文件: {log_file}")
    logging.info(f"保存伪标签: {args.save_pseudo_labels}")
    logging.info(f"使用CRF: {args.use_crf}")
    logging.info(f"胶囊网络参数: use_capsule={args.use_capsule}, disable_capsule={args.disable_capsule}")
    logging.info("="*50)

    start_time = datetime.datetime.now()
    
    results = main(cfg=cfg)
    
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    
    logging.info("="*50)
    logging.info("伪标签评估程序完成")
    logging.info(f"开始时间: {start_time}")
    logging.info(f"结束时间: {end_time}")
    logging.info(f"总耗时: {elapsed_time}")
    logging.info("="*50)

    print("\n=== 最终评估结果 ===")
    for eval_name, score in results.items():
        print(f"{eval_name.upper()}: mIoU = {score['miou']:.4f}") 