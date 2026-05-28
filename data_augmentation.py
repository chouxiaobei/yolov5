import os
import json
import numpy as np
from PIL import Image
import albumentations as A
import cv2

# 设置路径 - 使用正斜杠避免编码问题
base_path = r"E:\grade_based_training\datasets"
source_dir = base_path + "/traffic"
output_dir = base_path + "/augmented"

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)
print(f"源文件夹: {source_dir}")
print(f"输出文件夹: {output_dir}\n")

# 定义多种数据增强策略
def get_augmentations():
    """定义多种数据增强组合"""
    
    augmentations = [
        {
            'name': 'horizontal_flip',
            'transform': A.Compose([
                A.HorizontalFlip(p=1.0),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])),
            'description': '水平翻转'
        },
        {
            'name': 'brightness_contrast',
            'transform': A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])),
            'description': '亮度和对比度调整'
        },
        {
            'name': 'rotation',
            'transform': A.Compose([
                A.Rotate(
                    limit=30,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])),
            'description': '随机旋转（±30度）'
        },
        # {
        #     'name': 'gaussian_noise',
        #     'transform': A.Compose([
        #         A.GaussNoise(
        #             std_range=(0.1, 0.2),
        #             p=1.0
        #         ),
        #     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])),
        #     'description': '高斯噪声'
        # },
        # {
        #     'name': 'random_crop',
        #     'transform': A.Compose([
        #         A.RandomResizedCrop(
        #             size=(800, 800),
        #             scale=(0.8, 1.0),
        #             ratio=(0.9, 1.1),
        #             p=1.0
        #         ),
        #     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])),
        #     'description': '随机裁剪（保持80%-100%区域）'
        # },
        # {
        #     'name': 'affine_transform',
        #     'transform': A.Compose([
        #         A.Affine(
        #             scale=(0.8, 1.2),
        #             translate_percent=(-0.1, 0.1),
        #             rotate=(-15, 15),
        #             shear=(-10, 10),
        #             interpolation=cv2.INTER_LINEAR,
        #             p=1.0
        #         ),
        #     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])),
        #     'description': '仿射变换（缩放、平移、旋转、剪切）'
        # },
    ]
    
    return augmentations

def convert_labelme_to_albumentations(shapes, image_width, image_height):
    """将LabelMe格式的标注转换为Albumentations需要的格式"""
    bboxes = []
    labels = []
    
    for shape in shapes:
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            # 确保坐标有效性：左上角 < 右下角
            x_min = min(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            x_max = max(points[0][0], points[1][0])
            y_max = max(points[0][1], points[1][1])
            
            # 检查边界
            if x_min >= x_max or y_min >= y_max:
                print(f"跳过无效标注: {shape['label']}")
                continue
            
            # Albumentations需要的是 [x_min, y_min, x_max, y_max] 格式
            bbox = [x_min, y_min, x_max, y_max]
            bboxes.append(bbox)
            labels.append(shape['label'])
    
    return bboxes, labels

def convert_albumentations_to_labelme(bboxes, labels, transformed_image_shape, original_shapes):
    """将Albumentations转换后的bbox转回LabelMe格式"""
    new_shapes = []
    
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x_min, y_min, x_max, y_max = bbox
        
        # 确保坐标在图像范围内
        img_height, img_width = transformed_image_shape[:2]
        x_min = max(0, min(x_min, img_width - 1))
        y_min = max(0, min(y_min, img_height - 1))
        x_max = max(0, min(x_max, img_width))
        y_max = max(0, min(y_max, img_height))
        
        # 确保标注有效
        if x_min >= x_max or y_min >= y_max:
            continue
        
        # 创建LabelMe格式的shape
        new_shape = {
            'label': label,
            'points': [
                [float(x_min), float(y_min)],
                [float(x_max), float(y_max)]
            ],
            'group_id': None,
            'description': '',
            'shape_type': 'rectangle',
            'flags': {},
            'mask': None
        }
        new_shapes.append(new_shape)
    
    return new_shapes

def augment_image(image_path, json_path, output_dir, aug_config):
    """对单张图片进行数据增强"""
    
    try:
        # 读取原始JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 使用PIL读取图片，然后转换为numpy数组供albumentations使用
        pil_image = Image.open(image_path).convert('RGB')
        image = np.array(pil_image)
        
        if image is None or image.size == 0:
            print(f"无法读取图片: {os.path.basename(image_path)}")
            return False
        
        original_height, original_width = image.shape[:2]
        
        # 提取标注
        shapes = json_data['shapes']
        bboxes, labels = convert_labelme_to_albumentations(shapes, original_width, original_height)
        
        if len(bboxes) == 0:
            print(f"没有有效的标注框")
            return False
        
        # 应用数据增强
        transform = aug_config['transform']
        transformed = transform(
            image=image,
            bboxes=bboxes,
            category_ids=list(range(len(labels)))
        )
        
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        
        if len(transformed_bboxes) == 0:
            print(f"增强后没有保留标注框")
            return False
        
        # 转换回LabelMe格式
        new_shapes = convert_albumentations_to_labelme(
            transformed_bboxes, 
            labels, 
            transformed_image.shape,
            shapes
        )
        
        if len(new_shapes) == 0:
            print(f"转换后没有有效的标注")
            return False
        
        # 创建新的JSON数据
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        ext = os.path.splitext(base_name)[1]
        
        new_image_name = f"{name_without_ext}_{aug_config['name']}{ext}"
        new_json_name = f"{name_without_ext}_{aug_config['name']}.json"
        
        new_json_data = {
            'version': json_data['version'],
            'flags': json_data.get('flags', {}),
            'shapes': new_shapes,
            'imagePath': new_image_name,
            'imageData': None,
            'imageHeight': int(transformed_image.shape[0]),
            'imageWidth': int(transformed_image.shape[1])
        }
        
        # 保存增强后的图片（使用PIL保存，避免OpenCV的中文路径问题）
        augmented_pil = Image.fromarray(transformed_image)
        output_image_path = os.path.join(output_dir, new_image_name)
        augmented_pil.save(output_image_path, quality=95)
        
        # 保存新的JSON文件
        output_json_path = os.path.join(output_dir, new_json_name)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_json_data, f, ensure_ascii=False, indent=2)
        
        print(f"{aug_config['description']}: {new_image_name} ({len(new_shapes)}个标注)")
        return True
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个标注文件\n")
    
    # 获取增强配置
    augmentations = get_augmentations()
    print(f"可用的增强策略 ({len(augmentations)}种):")
    for i, aug in enumerate(augmentations, 1):
        print(f"  {i}. {aug['name']}: {aug['description']}")
    print()
    
    # 统计信息
    total_success = 0
    total_fail = 0
    
    # 对每张图片应用所有增强策略
    for json_file in sorted(json_files):
        base_name = json_file.replace('.json', '')
        
        # 查找对应的图片文件
        image_file = None
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            test_path = os.path.join(source_dir, base_name + ext)
            if os.path.exists(test_path):
                image_file = test_path
                break
        
        if image_file is None:
            print(f"未找到图片: {base_name}")
            total_fail += 1
            continue
        
        print(f"\n处理: {base_name}")
        
        # 对每种增强策略进行处理
        for aug_config in augmentations:
            success = augment_image(
                image_file, 
                os.path.join(source_dir, json_file),
                output_dir,
                aug_config
            )
            if success:
                total_success += 1
            else:
                total_fail += 1
    
    # 输出统计信息
    print("\n" + "="*70)
    print("数据增强完成！")
    print("="*70)
    print(f"成功生成: {total_success} 个增强样本")
    print(f"失败: {total_fail} 个")
    print(f"输出文件夹: {output_dir}")
    
    # 统计输出文件夹中的文件
    output_files = os.listdir(output_dir)
    output_images = [f for f in output_files if f.endswith(('.jpg', '.JPG', '.png', 'PNG'))]
    output_jsons = [f for f in output_files if f.endswith('.json')]
    
    print(f"\n输出文件夹内容:")
    print(f"  图片文件: {len(output_images)} 个")
    print(f"  JSON文件: {len(output_jsons)} 个")
    print(f"  总计: {len(output_files)} 个文件")
    
    print(f"所有生成的图片和JSON文件都可以用LabelMe打开复查")

if __name__ == '__main__':
    main()
