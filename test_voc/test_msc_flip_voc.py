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
parser.add_argument("--work_dir", default="results_capsule_6", type=str, help="work_dir")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=512, type=int, help="resize the long side")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set") #val
parser.add_argument("--model_path", default="../train_voc/experiment_capsule/checkpoints/2025-06-27-18-01/WeCLIP_capsule_model_iter_84000.pth", type=str, help="model_path")

# Capsule Network Arguments
parser.add_argument("--use_capsule", type=bool, default=True, help="Enable capsule network (default: False for testing, override if model was trained with capsules)")
parser.add_argument("--disable_capsule", action="store_true", help="Explicitly disable capsule network (overrides use_capsule if both are set)")
parser.add_argument("--primary_caps_num", default=32, type=int, help="Number of primary capsules")
parser.add_argument("--primary_caps_dim", default=8, type=int, help="Dimension of primary capsules")
parser.add_argument("--num_routing", default=3, type=int, help="Number of routing iterations")

def validate(model, dataset, test_scales=None):

    _preds, _gts, _msc_preds, cams = [], [], [], []
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    # with torch.no_grad(), torch.cuda.device(0):
    model.cuda(0)
    model.eval()
    
    logging.info(f"开始验证，数据集大小: {len(dataset)}, 测试尺度: {test_scales}")
    logging.info(f"开始验证，胶囊权重: {0.3}, 分割权重: {0.7}")

    num = 0

    _preds_hist = np.zeros((21, 21))
    _msc_preds_hist = np.zeros((21, 21))
    _cams_hist = np.zeros((21, 21))

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
        num+=1

        name, inputs, labels, cls_labels = data
        names = name+name

        inputs = inputs.cuda()
        labels = labels.cuda()

        #######
        # resize long side to 512
        
        _, _, h, w = inputs.shape
        ratio = args.resize_long / max(h,w)
        _h, _w = int(h*ratio), int(w*ratio)
        inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)
        
        #######

        segs_list = []
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        segs_cat, cam, attn_loss = model(inputs_cat, names, mode = 'val')
        
        cam = cam[0].unsqueeze(0)
        segs = segs_cat[0].unsqueeze(0)

        _segs = (segs_cat[0,...] + segs_cat[1,...].flip(-1)) / 2
        segs_list.append(_segs)

        _, _, h, w = segs_cat.shape

        for s in test_scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs_cat, cam_cat, attn_loss = model(inputs_cat, names, mode='val')

                _segs_cat = F.interpolate(segs_cat, size=(h, w), mode='bilinear', align_corners=False)
                _segs = (_segs_cat[0,...] + _segs_cat[1,...].flip(-1)) / 2
                segs_list.append(_segs)

        msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_preds = torch.argmax(resized_segs, dim=1)

        resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

        cams += list(cam.cpu().numpy().astype(np.int16))
        _preds += list(seg_preds.cpu().numpy().astype(np.int16))
        _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
        _gts += list(labels.cpu().numpy().astype(np.int16))


        if num % 100 == 0:
            _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist)
            _msc_preds_hist, msc_seg_score = evaluate.scores(_gts, _msc_preds, _msc_preds_hist)
            _cams_hist, cam_score = evaluate.scores(_gts, cams, _cams_hist)
            logging.info(f"已处理 {num} 张图像 - CAM mIoU: {cam_score['miou']:.4f}, Seg mIoU: {seg_score['miou']:.4f}, MSC Seg mIoU: {msc_seg_score['miou']:.4f}")
            _preds, _gts, _msc_preds, cams = [], [], [], []


        np.save(args.work_dir+ '/logit/' + name[0] + '.npy', {"segs":segs.detach().cpu().numpy(), "msc_segs":msc_segs.detach().cpu().numpy()})
    
    logging.info("验证完成")        
    return _gts, _preds, _msc_preds, cams, _preds_hist, _msc_preds_hist, _cams_hist

def crf_proc(config):
    print("crf post-processing...")
    logging.info("开始CRF后处理...")

    txt_name = os.path.join(config.dataset.name_list_dir, args.eval_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    logging.info(f"CRF处理图像数量: {len(name_list)}")

    images_path = os.path.join(config.dataset.root_dir, 'JPEGImages',)
    labels_path = os.path.join(config.dataset.root_dir, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=64,  # 64
        bi_rgb_std=5,   # 5
        bi_w=4,         # 4
    )

    def _job(i):

        name = name_list[i]
        logit_name = os.path.join(args.work_dir, "logit", name + ".npy")

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_segs']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.eval_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(os.path.join(args.work_dir, "prediction", name + ".png"), np.squeeze(pred).astype(np.uint8))
        imageio.imsave(os.path.join(args.work_dir, "prediction_cmap", name + ".png"), encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    n_jobs = int(multiprocessing.cpu_count() * 0.8)
    #results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])
    results = joblib.Parallel(n_jobs=4, verbose=10, pre_dispatch="all")(
        [joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)
    hist = np.zeros((21, 21))
    hist, score = evaluate.scores(gts, preds, hist, 21)

    logging.info("CRF后处理完成")
    logging.info(f"CRF后处理结果: {score}")
    print(score)
    
    return True

def main(cfg):
    
    logging.info("开始测试流程")
    logging.info("新的加了diceloss的网络，权重0.3，0.7")
    logging.info(f"配置信息: {cfg}")
    logging.info(f"参数信息: {args}")
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    logging.info(f"验证数据集加载完成，数据集大小: {len(val_dataset)}")

    # 处理胶囊网络参数逻辑
    actual_use_capsule = args.use_capsule
    if args.disable_capsule:
        actual_use_capsule = False

    capsule_config = None
    if actual_use_capsule:
        capsule_config = {
            'feature_size': cfg.dataset.crop_size // 16, # 根据配置文件中的crop_size或默认值
            'primary_caps_num': args.primary_caps_num,
            'primary_caps_dim': args.primary_caps_dim,
            'class_caps_dim': 16, # 假设与训练时一致
            'num_routing': args.num_routing,
            'enable_segmentation': True, # 假设与训练时一致
            'enable_feature_enhancement': True # 假设与训练时一致
        }
        logging.info(f"启用胶囊网络进行测试，配置: {capsule_config}")
    else:
        logging.info("未启用胶囊网络进行测试")

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

    gts, preds, msc_preds, cams, preds_hist, msc_preds_hist, cams_hist = validate(model=WeCLIP_model, dataset=val_dataset, test_scales=[1, 0.75])
    torch.cuda.empty_cache()

    preds_hist, seg_score = evaluate.scores(gts, preds, preds_hist)
    msc_preds_hist, msc_seg_score = evaluate.scores(gts, msc_preds, msc_preds_hist)
    cams_hist, cam_score = evaluate.scores(gts, cams, cams_hist)

    logging.info("验证完成，最终结果:")
    logging.info(f"CAM分数: {cam_score}")
    logging.info(f"分割分数: {seg_score}")
    logging.info(f"多尺度分割分数: {msc_seg_score}")

    print("cams score:")
    print(cam_score)
    print("segs score:")
    print(seg_score)
    print("msc segs score:")
    print(msc_seg_score)

    crf_proc(config=cfg)

    logging.info("测试流程完成")
    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    print(cfg)
    print(args)

    args.work_dir = os.path.join(args.work_dir, args.eval_set)

    os.makedirs(args.work_dir + "/logit", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_cmap", exist_ok=True)
    
    # 设置日志文件
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    log_file = os.path.join(args.work_dir, f"test_log_{timestamp}.log")
    setup_logger(log_file)
    
    logging.info("="*50)
    logging.info("开始测试程序")
    logging.info(f"时间戳: {timestamp}")
    logging.info(f"模型路径: {args.model_path}")
    logging.info(f"评估集: {args.eval_set}")
    logging.info(f"工作目录: {args.work_dir}")
    logging.info(f"日志文件: {log_file}")
    logging.info(f"胶囊网络参数: use_capsule={args.use_capsule}, primary_caps_num={args.primary_caps_num}, primary_caps_dim={args.primary_caps_dim}, num_routing={args.num_routing}, disable_capsule={args.disable_capsule}")
    logging.info("="*50)

    start_time = datetime.datetime.now()
    
    main(cfg=cfg)
    # crf_proc(config=cfg)
    
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    
    logging.info("="*50)
    logging.info("测试程序完成")
    logging.info(f"开始时间: {start_time}")
    logging.info(f"结束时间: {end_time}")
    logging.info(f"总耗时: {elapsed_time}")
    logging.info("="*50)
