import numpy as np
from numpy.lib.utils import deprecate
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio.v2 as imageio
from . import transforms
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list

def load_cls_label_list(name_list_dir):
    
    return np.load(os.path.join(name_list_dir,'cls_labels_onehot.npy'), allow_pickle=True).item()


class VOC12Dataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        name_list_dir=None,
        split='train',
        stage='train',
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.label_dir = os.path.join(root_dir, 'SegmentationClassAug')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name+'.jpg')
        image = np.asarray(imageio.imread(img_name))
        # image = Image.open(img_name)

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "test":
            label = image[:,:,0]

        return _img_name, image, label

def _transform_resize():
    return Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class VOC12ClsDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
        self.normalize = _transform_resize()
        self.scale = 1
        self.patch_size = 16

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image):
        img_box = None
        if self.aug:
            image = np.array(image)
            # print('image', image.shape)
            '''
            if self.resize_range: 
                image, label = transforms.random_resize(
                    image, label, size_range=self.resize_range)
            '''
            if self.rescale_range:
                image = transforms.random_scaling(
                    image,
                    scale_range=self.rescale_range)
            
            if self.img_fliplr:
                image = transforms.random_fliplr(image)
            #image = self.color_jittor(image)
            if self.crop_size:
                image, img_box = transforms.random_crop(
                    image,
                    crop_size=self.crop_size,
                    mean_rgb=[0,0,0],#[123.675, 116.28, 103.53], 
                    ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
        image = transforms.normalize_img(image)
        # image = self.normalize(image)
        # image = image.numpy()
        
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, img_box

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        #label_onehot = F.one_hot(label, num_classes)
        
        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):

        img_name, image, _ = super().__getitem__(idx)
        
#         ori_height = image.size[1]
#         ori_width = image.size[0]
        
#         new_height = int(np.ceil(self.scale * int(ori_height) / self.patch_size) * self.patch_size)
#         new_width = int(np.ceil(self.scale * int(ori_width) / self.patch_size) * self.patch_size)
        
#         image = Resize((new_height, new_width), interpolation=BICUBIC)(image)
#         image = image.convert("RGB")

        image, img_box = self.__transforms(image=image)

        cls_label = self.label_list[img_name]

        if self.aug:
            return img_name, image, cls_label, img_box
        else:
            return img_name, image, cls_label


class VOC12SegDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 pseudo_label_dir=None,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()
        self.normalize = _transform_resize()
        self.pseudo_label_dir = pseudo_label_dir

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
        self.scale = 1
        self.patch_size = 16
        

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        img_box = None
        if self.aug:
            image = np.array(image)
            '''
            if self.resize_range: 
                image, label = transforms.random_resize(
                    image, label, size_range=self.resize_range)
            
            if self.rescale_range:
                image, label = transforms.random_scaling(
                    image,
                    label,
                    scale_range=self.rescale_range)
            '''
            if self.img_fliplr:
                image, label = transforms.random_fliplr(image, label)
            if self.pseudo_label_dir is None:
                image = self.color_jittor(image)
            if self.crop_size:
                image, label, img_box = transforms.random_crop(
                    image,
                    label,
                    crop_size=self.crop_size,
                    mean_rgb=[0,0,0] if self.pseudo_label_dir else None,
                    ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
        # image = self.normalize(image)
        # image = image.numpy()
        
        image = transforms.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label, img_box

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)
        
        if self.pseudo_label_dir is not None:
            pseudo_label_path = os.path.join(self.pseudo_label_dir, img_name + '.png')
            if os.path.exists(pseudo_label_path):
                label = np.asarray(imageio.imread(pseudo_label_path))
            else:
                label = np.ones_like(image[:, :, 0]) * self.ignore_index
        
#         ori_height = image.size[1]
#         ori_width = image.size[0]
        
#         new_height = int(np.ceil(self.scale * int(ori_height) / self.patch_size) * self.patch_size)
#         new_width = int(np.ceil(self.scale * int(ori_width) / self.patch_size) * self.patch_size)
        
#         image = Resize((new_height, new_width), interpolation=BICUBIC)(image)
#         image = image.convert("RGB")

        image, label, img_box = self.__transforms(image=image, label=label)

        if self.stage=='test':
            cls_label = 0
        else:
            cls_label = self.label_list[img_name]

        if self.pseudo_label_dir is not None and self.aug:
            return img_name, image, label, cls_label, img_box
        else:
            return img_name, image, label, cls_label


class VOC12PseudoLabelDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=320,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
                 pseudo_label_dir=None,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()
        self.pseudo_label_dir = pseudo_label_dir

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
        self.normalize = _transform_resize()
        self.scale = 1
        self.patch_size = 16

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, pseudo_label):
        img_box = None
        if self.aug:
            image = np.array(image)
            if self.rescale_range:
                image, pseudo_label = transforms.random_scaling(
                    image,
                    pseudo_label,
                    scale_range=self.rescale_range)

            if self.img_fliplr:
                image, pseudo_label = transforms.random_fliplr(image, pseudo_label)

            if self.crop_size:
                image, pseudo_label, img_box = transforms.random_crop(
                    image,
                    pseudo_label,
                    crop_size=self.crop_size,
                    mean_rgb=[0,0,0],  # 与VOC12ClsDataset保持一致
                    ignore_index=self.ignore_index)

        image = transforms.normalize_img(image)
        # 转换图像格式为CHW
        image = np.transpose(image, (2, 0, 1))

        return image, pseudo_label, img_box

    def __getitem__(self, idx):
        img_name, image, _ = super().__getitem__(idx)

        # 加载伪标签
        if self.pseudo_label_dir is not None:
            pseudo_label_path = os.path.join(self.pseudo_label_dir, img_name + '.png')
            # print("伪标签文件路径:", pseudo_label_path)
            # print(os.path.exists(pseudo_label_path))
            # input()
            if os.path.exists(pseudo_label_path):
                pseudo_label = np.asarray(imageio.imread(pseudo_label_path))
            else:
                # 如果伪标签不存在，创建一个全为ignore_index的标签
                print("伪标签不存在不存在")
                pseudo_label = np.ones_like(image[:, :, 0]) * self.ignore_index
        else:
            # 如果未指定伪标签目录，创建一个全为ignore_index的标签
            pseudo_label = np.ones_like(image[:, :, 0]) * self.ignore_index

        image, pseudo_label, img_box = self.__transforms(image=image, pseudo_label=pseudo_label)

        cls_label = self.label_list[img_name]

        if self.aug:
            return img_name, image, pseudo_label, cls_label, img_box
        else:
            return img_name, image, pseudo_label, cls_label