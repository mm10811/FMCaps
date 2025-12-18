"""
COCO分割真值可视化脚本
将单通道灰度真值图转换为彩色可视化图像
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_coco_colormap(num_classes=81):
    """
    生成COCO数据集的颜色映射表
    Args:
        num_classes: 类别数量（COCO为81，包括背景）
    Returns:
        colormap: numpy数组，形状为 (num_classes, 3)
    """
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    
    for i in range(num_classes):
        r, g, b = 0, 0, 0
        c = i
        for j in range(8):
            r = r | ((c & 1) << (7 - j))
            g = g | (((c >> 1) & 1) << (7 - j))
            b = b | (((c >> 2) & 1) << (7 - j))
            c = c >> 3
        colormap[i] = [r, g, b]
    
    # 背景设为黑色
    colormap[0] = [0, 0, 0]
    
    return colormap


def get_coco_colormap_v2(num_classes=81):
    """
    生成更鲜艳的COCO数据集颜色映射表
    使用HSV色彩空间生成均匀分布的颜色
    """
    import colorsys
    
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]  # 背景为黑色
    
    for i in range(1, num_classes):
        # 使用HSV生成均匀分布的颜色
        hue = (i - 1) / (num_classes - 1)
        saturation = 0.8 + 0.2 * ((i % 3) / 2)  # 饱和度在0.8-1.0之间变化
        value = 0.7 + 0.3 * ((i % 5) / 4)  # 亮度在0.7-1.0之间变化
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colormap[i] = [int(r * 255), int(g * 255), int(b * 255)]
    
    return colormap


def apply_colormap(gray_image, colormap):
    """
    将灰度图应用颜色映射
    Args:
        gray_image: 灰度图像，numpy数组
        colormap: 颜色映射表
    Returns:
        color_image: 彩色图像
    """
    h, w = gray_image.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(colormap)):
        mask = gray_image == class_id
        color_image[mask] = colormap[class_id]
    
    # 处理ignore_index (255)
    ignore_mask = gray_image == 255
    color_image[ignore_mask] = [128, 128, 128]  # 灰色表示忽略区域
    
    return color_image


def visualize_single_image(input_path, output_path, colormap):
    """
    可视化单张图像
    """
    # 读取灰度图
    gray_img = np.array(Image.open(input_path))
    
    # 应用颜色映射
    color_img = apply_colormap(gray_img, colormap)
    
    # 保存彩色图
    Image.fromarray(color_img).save(output_path)


def visualize_directory(input_dir, output_dir, colormap, num_samples=None):
    """
    可视化整个目录的图像
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        colormap: 颜色映射表
        num_samples: 最大处理样本数，None表示处理所有
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有png文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    if num_samples is not None:
        image_files = image_files[:num_samples]
    
    print(f"共找到 {len(image_files)} 张图像")
    
    for img_name in tqdm(image_files, desc="处理中"):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        
        try:
            visualize_single_image(input_path, output_path, colormap)
        except Exception as e:
            print(f"处理 {img_name} 失败: {e}")


def create_legend(colormap, class_names=None, output_path="coco_legend.png"):
    """
    创建颜色图例
    Args:
        colormap: 颜色映射表
        class_names: 类别名称列表
        output_path: 输出路径
    """
    num_classes = len(colormap)
    
    # 默认COCO类别名称
    if class_names is None:
        class_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    # 计算图例尺寸
    cols = 4
    rows = (num_classes + cols - 1) // cols
    cell_width = 200
    cell_height = 30
    color_box_size = 20
    
    legend_width = cols * cell_width
    legend_height = rows * cell_height + 50  # 额外空间用于标题
    
    # 创建图例图像
    legend = Image.new('RGB', (legend_width, legend_height), (255, 255, 255))
    
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(legend)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    # 绘制标题
    draw.text((legend_width // 2 - 100, 10), "COCO Segmentation Classes", fill=(0, 0, 0), font=title_font)
    
    # 绘制每个类别
    for i in range(min(num_classes, len(class_names))):
        col = i % cols
        row = i // cols
        
        x = col * cell_width + 10
        y = row * cell_height + 50
        
        # 绘制颜色框
        color = tuple(colormap[i])
        draw.rectangle([x, y, x + color_box_size, y + color_box_size], fill=color, outline=(0, 0, 0))
        
        # 绘制类别名称
        text = f"{i}: {class_names[i]}"
        draw.text((x + color_box_size + 5, y + 3), text, fill=(0, 0, 0), font=font)
    
    legend.save(output_path)
    print(f"图例已保存到: {output_path}")


def analyze_image(input_path):
    """
    分析图像中的类别分布
    """
    gray_img = np.array(Image.open(input_path))
    unique_classes = np.unique(gray_img)
    
    print(f"图像: {input_path}")
    print(f"尺寸: {gray_img.shape}")
    print(f"类别数: {len(unique_classes)}")
    print(f"类别列表: {unique_classes}")
    
    for cls in unique_classes:
        if cls == 255:
            print(f"  ignore (255): {(gray_img == cls).sum()} 像素")
        else:
            print(f"  class {cls}: {(gray_img == cls).sum()} 像素")


def main():
    # ==================== 内置参数配置 ====================
    # 输入路径（单张图像或目录）
    input_path = "../MSCOCO/SegmentationClass/val"
    # input_path = "../MSCOCO/SegmentationClass/val"
    # input_path = "../MSCOCO/SegmentationClass/train/000000000009.png"  # 单张图像
    
    # 输出路径（None表示自动生成）
    output_path = "../MSCOCO/SegmentationClass_colored/val"
    # output_path = "../MSCOCO/SegmentationClass_colored/val"
    # output_path = None  # 自动生成
    
    # 颜色映射方案: 'v1'(标准), 'v2'(鲜艳)
    colormap_type = 'v1'
    
    # 最大处理样本数，None表示处理所有
    num_samples = None
    # num_samples = 100  # 只处理100张
    
    # 是否生成颜色图例
    generate_legend = False
    
    # 是否分析图像类别分布
    analyze_mode = False
    
    # 类别数量（COCO为81）
    num_classes = 81
    # ======================================================
    
    # 选择颜色映射
    if colormap_type == 'v1':
        colormap = get_coco_colormap(num_classes)
    else:
        colormap = get_coco_colormap_v2(num_classes)
    
    # 生成图例
    if generate_legend:
        # 设置图例保存路径
        if output_path and output_path.endswith('.png'):
            legend_path = output_path
        else:
            # 默认保存到当前目录
            legend_path = "coco_legend.png"
        print(f"生成颜色图例: {legend_path}")
        create_legend(colormap, output_path=legend_path)
        return
    
    # 分析模式
    if analyze_mode:
        if os.path.isfile(input_path):
            analyze_image(input_path)
        else:
            # 分析目录中的第一张图
            files = [f for f in os.listdir(input_path) if f.endswith('.png')]
            if files:
                analyze_image(os.path.join(input_path, files[0]))
        return
    
    # 设置输出路径
    if output_path is None:
        if os.path.isfile(input_path):
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_color{ext}"
        else:
            output_path = input_path + "_colored"
    
    # 处理单张图像或目录
    if os.path.isfile(input_path):
        print(f"处理单张图像: {input_path}")
        visualize_single_image(input_path, output_path, colormap)
        print(f"已保存到: {output_path}")
    else:
        print(f"处理目录: {input_path}")
        visualize_directory(input_path, output_path, colormap, num_samples)
        print(f"已保存到: {output_path}")


if __name__ == "__main__":
    main()

