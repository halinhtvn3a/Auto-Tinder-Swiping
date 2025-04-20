"""
Script để cân bằng dữ liệu trong dataset giữa lớp 'like' và 'dislike'
bằng cách sử dụng data augmentation hoặc undersampling
python balance_dataset.py --method both --ratio 1.0
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import shutil
import random
from tqdm import tqdm
from PreferenceRecognition.data_collector import DataCollector

def augment_image(image, rotation=10, flip=True, brightness=0.2, contrast=0.2):
    """
    Tạo phiên bản tăng cường của ảnh
    
    Args:
        image: Ảnh đầu vào
        rotation: Góc xoay tối đa
        flip: Có lật ảnh không
        brightness: Mức điều chỉnh độ sáng
        contrast: Mức điều chỉnh độ tương phản
        
    Returns:
        augmented_img: Ảnh đã tăng cường
    """
    img = image.copy()
    
    # Xoay ảnh ngẫu nhiên
    if rotation > 0:
        angle = random.uniform(-rotation, rotation)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Lật ảnh
    if flip and random.random() > 0.5:
        img = cv2.flip(img, 1)  # 1 = lật ngang
    
    # Điều chỉnh độ sáng
    if brightness > 0:
        beta = random.uniform(-brightness*255, brightness*255)
        img = cv2.convertScaleAbs(img, beta=beta)
    
    # Điều chỉnh độ tương phản
    if contrast > 0:
        alpha = 1.0 + random.uniform(-contrast, contrast)
        img = cv2.convertScaleAbs(img, alpha=alpha)
        
    return img

def balance_dataset(method='augment', target_ratio=1.0, max_augmentations=5):
    """
    Cân bằng dataset bằng cách tạo thêm dữ liệu cho lớp thiểu số
    hoặc giảm dữ liệu cho lớp đa số
    
    Args:
        method: Phương pháp cân bằng ('augment', 'undersample', 'both')
        target_ratio: Tỉ lệ mong muốn giữa lớp thiểu số và lớp đa số (1.0 = cân bằng hoàn toàn)
        max_augmentations: Số lượng tối đa augmentations cho mỗi ảnh
    """
    # Khởi tạo DataCollector để lấy đường dẫn thư mục
    data_collector = DataCollector()
    like_dir = data_collector.like_dir
    dislike_dir = data_collector.dislike_dir
    
    # Tìm tất cả file ảnh
    like_files = list(Path(like_dir).glob('*.jpg'))
    dislike_files = list(Path(dislike_dir).glob('*.jpg'))
    
    like_count = len(like_files)
    dislike_count = len(dislike_files)
    
    print(f"Số ảnh hiện tại: Like = {like_count}, Dislike = {dislike_count}")
    print(f"Tỉ lệ hiện tại: Like:Dislike = 1:{dislike_count/like_count:.2f}")
    
    # Xác định lớp thiểu số và lớp đa số
    if like_count < dislike_count:
        minority_class = "like"
        minority_files = like_files
        minority_count = like_count
        majority_class = "dislike"
        majority_files = dislike_files
        majority_count = dislike_count
    else:
        minority_class = "dislike"
        minority_files = dislike_files
        minority_count = dislike_count
        majority_class = "like"
        majority_files = like_files
        majority_count = like_count
    
    # Tính toán số lượng mục tiêu
    target_minority_count = minority_count
    target_majority_count = int(minority_count / target_ratio)
    
    if method in ['augment', 'both']:
        # Tăng cường dữ liệu cho lớp thiểu số
        print(f"Tăng cường dữ liệu cho lớp '{minority_class}'...")
        
        target_count = int(majority_count * target_ratio)
        augmentations_needed = target_count - minority_count
        
        if augmentations_needed <= 0:
            print(f"Không cần tăng cường dữ liệu cho lớp '{minority_class}'")
        else:
            # Số lượng augmentations cần tạo cho mỗi ảnh
            augmentations_per_image = min(
                max_augmentations,
                int(np.ceil(augmentations_needed / minority_count))
            )
            
            # Tạo thư mục tạm để lưu ảnh augmented
            minority_aug_dir = os.path.join(os.path.dirname(minority_files[0]), f"{minority_class}_augmented")
            os.makedirs(minority_aug_dir, exist_ok=True)
            
            print(f"Tạo {augmentations_needed} ảnh tăng cường... ({augmentations_per_image} phiên bản/ảnh)")
            
            created_count = 0
            for file in tqdm(minority_files):
                # Đọc ảnh
                img = cv2.imread(str(file))
                if img is None:
                    continue
                
                # Tạo các phiên bản augmented
                for i in range(augmentations_per_image):
                    if created_count >= augmentations_needed:
                        break
                    
                    # Augment ảnh
                    augmented = augment_image(img)
                    
                    # Tạo tên file mới
                    base_name = os.path.basename(file)
                    name_parts = os.path.splitext(base_name)
                    new_name = f"{name_parts[0]}_aug{i+1}{name_parts[1]}"
                    
                    # Lưu ảnh
                    output_path = os.path.join(minority_aug_dir, new_name)
                    cv2.imwrite(output_path, augmented)
                    created_count += 1
            
            # Di chuyển các file augmented vào thư mục chính
            aug_files = list(Path(minority_aug_dir).glob('*.jpg'))
            for file in aug_files:
                dest_path = os.path.join(os.path.dirname(minority_files[0]), os.path.basename(file))
                shutil.move(str(file), dest_path)
            
            # Xóa thư mục tạm
            os.rmdir(minority_aug_dir)
            
            print(f"Đã tạo {created_count} ảnh tăng cường cho lớp '{minority_class}'")
    
    if method in ['undersample', 'both']:
        # Giảm dữ liệu cho lớp đa số
        print(f"Giảm dữ liệu cho lớp '{majority_class}'...")
        
        if method == 'both':
            # Nếu đã tăng cường dữ liệu, tính lại số lượng cần giảm
            minority_files = list(Path(os.path.dirname(minority_files[0])).glob('*.jpg'))
            minority_count = len(minority_files)
            target_majority_count = int(minority_count / target_ratio)
        
        files_to_remove = majority_count - target_majority_count
        
        if files_to_remove <= 0:
            print(f"Không cần giảm dữ liệu cho lớp '{majority_class}'")
        else:
            # Tạo thư mục backup
            backup_dir = os.path.join(os.path.dirname(majority_files[0]), f"{majority_class}_backup")
            os.makedirs(backup_dir, exist_ok=True)
            
            print(f"Di chuyển {files_to_remove} ảnh từ lớp '{majority_class}' sang thư mục backup...")
            
            # Chọn ngẫu nhiên các file để xóa
            files_to_backup = random.sample(majority_files, files_to_remove)
            
            for file in tqdm(files_to_backup):
                dest_path = os.path.join(backup_dir, os.path.basename(file))
                shutil.move(str(file), dest_path)
            
            print(f"Đã di chuyển {files_to_remove} ảnh vào {backup_dir}")
    
    # Hiển thị thống kê cuối cùng
    like_files = list(Path(like_dir).glob('*.jpg'))
    dislike_files = list(Path(dislike_dir).glob('*.jpg'))
    
    like_count = len(like_files)
    dislike_count = len(dislike_files)
    
    print(f"\nSố ảnh sau khi cân bằng: Like = {like_count}, Dislike = {dislike_count}")
    print(f"Tỉ lệ mới: Like:Dislike = 1:{dislike_count/like_count:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Công cụ cân bằng dataset")
    parser.add_argument('--method', type=str, default='both', choices=['augment', 'undersample', 'both'],
                      help='Phương pháp cân bằng: tăng cường dữ liệu (augment), giảm dữ liệu (undersample), hoặc cả hai (both)')
    parser.add_argument('--ratio', type=float, default=1.0,
                      help='Tỉ lệ mong muốn giữa lớp thiểu số và lớp đa số (mặc định: 1.0 = cân bằng hoàn toàn)')
    parser.add_argument('--max-augmentations', type=int, default=5,
                      help='Số lượng tối đa phiên bản tăng cường cho mỗi ảnh của lớp thiểu số')
    
    args = parser.parse_args()
    
    balance_dataset(method=args.method, target_ratio=args.ratio, max_augmentations=args.max_augmentations)