import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import new_class_names, BACKGROUND_CATEGORY
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP_model.PAR import PAR
from .capsule_module import create_capsule_module




def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


class WeCLIP(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, embedding_dim=256, in_channels=512, 
                 dataset_root_path=None, device='cuda', use_capsule=False, capsule_config=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_capsule = use_capsule

        self.encoder, _ = clip.load(clip_model, device=device)

        for name, param in self.encoder.named_parameters():
            if "11" not in name:
                param.requires_grad=False

        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=11)
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)

        # 胶囊网络模块 (可选)
        if self.use_capsule:
            if capsule_config is None:
                capsule_config = {
                    'feature_size': 20,
                    'primary_caps_num': 32,
                    'primary_caps_dim': 8,
                    'class_caps_dim': 16,
                    'num_routing': 3,
                    'enable_segmentation': True,
                    'enable_feature_enhancement': True
                }
            
            self.capsule_module = create_capsule_module(
                in_channels=self.embedding_dim,
                num_classes=self.num_classes,
                **capsule_config
            )
            
            # 特征融合权重
            self.capsule_weight = nn.Parameter(torch.tensor(0.3))  # 胶囊网络权重
            self.original_weight = nn.Parameter(torch.tensor(0.7))  # 原始网络权重

        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], self.encoder)
        self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], self.encoder)

        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform)
        self.root_path = os.path.join(dataset_root_path, 'SegmentationClassAug')
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
        self.iter_num = 0
        self.require_all_fts = True


    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        
        # 添加胶囊网络参数组
        if self.use_capsule:
            capsule_params = []
            for param in list(self.capsule_module.parameters()):
                capsule_params.append(param)
            capsule_params.extend([self.capsule_weight, self.original_weight])
            param_groups.append(capsule_params)

        return param_groups
    


    def forward(self, img, img_names='2007_000032', mode='train'):
        cam_list = []
        b, c, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True)

        fts_all_stack = torch.stack(fts_all, dim=0) # (11, hw, b, c)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        if self.require_all_fts==True:
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3) #(1, hw, 1, c)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//16, w //16) #(11, b, c, h, w)

        fts = self.decoder_fts_fuse(all_img_tokens)
        attn_fts = fts.clone()
        _, _, fts_h, fts_w = fts.shape
        
        # 胶囊网络处理 (如果启用)
        capsule_outputs = None
        if self.use_capsule:
            capsule_outputs, enhanced_fts = self.capsule_module(fts, return_enhanced_features=True)
            # 使用增强特征进行后续处理
            fts = enhanced_fts
        
        seg, seg_attn_weight_list = self.decoder(fts)
        
        # 如果使用胶囊网络，融合分割结果
        if self.use_capsule and 'seg_logits' in capsule_outputs:
            capsule_seg = capsule_outputs['seg_logits']
            # 上采样胶囊分割结果到与原始分割相同尺寸
            if capsule_seg.shape[-2:] != seg.shape[-2:]:
                capsule_seg = F.interpolate(capsule_seg, size=seg.shape[-2:], 
                                          mode='bilinear', align_corners=False)
            
            # 加权融合
            seg = torch.sigmoid(self.original_weight) * seg + torch.sigmoid(self.capsule_weight) * capsule_seg

        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.root_path, str(img_name)+'.png')
            img_i = img[i]
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]
            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]

            if self.iter_num > 15000 or mode=='val': #15000
                require_seg_trans = True
            else:
                require_seg_trans = False

            cam_refined_list, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                   self.bg_text_features, self.fg_text_features,
                                                                   self.grad_cam,
                                                                   mode=mode,
                                                                   require_seg_trans=require_seg_trans)

            cam_dict = generate_cam_label(cam_refined_list, keys, w, h)

            cams = cam_dict['refined_cam'].cuda()

            bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()
            cams = torch.cat([bg_score, cams], dim=0).cuda()

            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).cuda()

            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)

            cam_list.append(cam_labels)

        all_cam_labels = torch.stack(cam_list, dim=0)

        # 根据模式返回不同的输出
        if mode == 'train' and self.use_capsule and capsule_outputs is not None:
            return seg, all_cam_labels, attn_pred, capsule_outputs
        else:
            return seg, all_cam_labels, attn_pred
    
    def compute_capsule_loss(self, capsule_outputs, labels):
        """
        计算胶囊网络损失
        Args:
            capsule_outputs: 胶囊网络输出字典
            labels: 真实标签 [batch_size, height, width]
        Returns:
            capsule_loss: 胶囊网络损失
            loss_dict: 损失详情字典
        """
        if not self.use_capsule or capsule_outputs is None:
            return torch.tensor(0.0, device=labels.device), {}
        
        return self.capsule_module.compute_loss(capsule_outputs, labels)
    
    def set_capsule_enabled(self, enabled=True):
        """
        动态启用/禁用胶囊网络
        Args:
            enabled: 是否启用胶囊网络
        """
        self.use_capsule = enabled
        if hasattr(self, 'capsule_module'):
            for param in self.capsule_module.parameters():
                param.requires_grad = enabled

        
