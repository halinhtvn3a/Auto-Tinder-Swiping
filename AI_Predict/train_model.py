"""
Script để huấn luyện mô hình nhận diện sở thích từ dữ liệu hình ảnh

Usage:
    python train_model.py [--epochs 30] [--batch-size 16] [--transfer-learning] [--base-network MobileNetV2]
"""

import os
import argparse
import matplotlib.pyplot as plt
from PreferenceRecognition.model import PreferenceModel
from PreferenceRecognition.data_collector import DataCollector
from PreferenceRecognition.image_processor import ImageProcessor
import cv2
import numpy as np
import random
from pathlib import Path
import tensorflow as tf

# Cấu hình GPU nếu có
def setup_gpu():
    """Thiết lập GPU cho TensorFlow"""
    try:
        # Kiểm tra và cấu hình GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Đã phát hiện {len(gpus)} GPU:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
                
            # Thiết lập memory growth để tránh lỗi OOM (Out of Memory)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Đã bật memory growth cho tất cả GPU")
            
            # Sử dụng mixed precision training để tăng tốc trên GPU
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Đã bật mixed precision (FP16) để tăng tốc độ")
            
            return True
        else:
            print("Không tìm thấy GPU, sẽ sử dụng CPU")
            return False
    except Exception as e:
        print(f"Lỗi khi thiết lập GPU: {str(e)}")
        return False

def main():
    # Thiết lập GPU
    use_gpu = setup_gpu()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình nhận diện sở thích')
    parser.add_argument('--epochs', type=int, default=30, help='Số epochs để huấn luyện')
    parser.add_argument('--batch-size', type=int, default=16, help='Kích thước batch')
    parser.add_argument('--transfer-learning', action='store_true', 
                      help='Sử dụng transfer learning')
    parser.add_argument('--base-network', type=str, default='MobileNetV2', 
                      choices=['MobileNetV2', 'VGG16', 'ResNet50'], 
                      help='Mạng cơ sở cho transfer learning')
    parser.add_argument('--show-samples', action='store_true',
                      help='Hiển thị các mẫu ngẫu nhiên từ dataset trước khi train')
    parser.add_argument('--early-stopping-patience', type=int, default=7,
                      help='Số epochs chờ đợi trước khi dừng sớm')
    args = parser.parse_args()
    
    # Khởi tạo model và data collector
    model = PreferenceModel()
    data_collector = DataCollector()
    
    # Xác định đường dẫn data
    data_dir = data_collector.data_dir
    
    # Kiểm tra data
    stats = data_collector.get_dataset_stats()
    print(f"Thống kê dataset:")
    print(f"- Số ảnh 'thích': {stats['like_count']}")
    print(f"- Số ảnh 'không thích': {stats['dislike_count']}")
    print(f"- Tổng số ảnh: {stats['total_count']}")
    print(f"- Cân bằng dữ liệu: {'Có' if stats['is_balanced'] else 'Không'}")
    
    if stats['like_count'] < 10 or stats['dislike_count'] < 10:
        print("\nCẢNH BÁO: Dataset có ít hình ảnh. Kết quả có thể không tốt.")
        print("Khuyến nghị: Thêm ít nhất 10 hình ảnh mỗi loại.")
        
        proceed = input("Bạn vẫn muốn tiếp tục huấn luyện? (y/n): ")
        if proceed.lower() != 'y':
            print("Hủy quá trình huấn luyện.")
            return
    
    # Hiển thị các mẫu từ dataset nếu được yêu cầu
    if args.show_samples:
        show_dataset_samples(data_dir, num_samples=5)
    
    print(f"\nBắt đầu huấn luyện mô hình với {args.epochs} epochs và batch size {args.batch_size}")
    print(f"Transfer learning: {'Có' if args.transfer_learning else 'Không'}")
    if args.transfer_learning:
        print(f"Mạng cơ sở: {args.base_network}")
    print(f"Sử dụng GPU: {'Có' if use_gpu else 'Không'}")
    
    # Xây dựng model
    model.build_model(transfer_learning=args.transfer_learning, base_network=args.base_network)
    
    # In tóm tắt mô hình
    model.model.summary()
    
    # Huấn luyện model
    try:
        history = model.train(
            training_dir=data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Hiển thị kết quả
        final_acc = history.history.get('accuracy', [-1])[-1]
        final_val_acc = history.history.get('val_accuracy', [-1])[-1]
        
        print("\nHuấn luyện hoàn tất!")
        print(f"Accuracy: {final_acc:.4f}")
        print(f"Validation Accuracy: {final_val_acc:.4f}")
        print(f"Model đã được lưu tại {model.model_path}")
        
        # Vẽ biểu đồ accuracy và loss
        plot_training_history(history)
        
        print("\nBạn có thể sử dụng 'python evaluate_model.py' để đánh giá chi tiết mô hình.")
        
    except Exception as e:
        print(f"Lỗi khi huấn luyện model: {e}")

def show_dataset_samples(data_dir, num_samples=5):
    """Hiển thị một số mẫu ngẫu nhiên từ dataset"""
    try:
        import matplotlib.pyplot as plt
        import random
        from pathlib import Path
        import cv2
        
        like_dir = os.path.join(data_dir, 'like')
        dislike_dir = os.path.join(data_dir, 'dislike')
        
        like_files = list(Path(like_dir).glob('*.png'))
        dislike_files = list(Path(dislike_dir).glob('*.png'))
        
        # Lấy mẫu ngẫu nhiên
        like_samples = random.sample(like_files, min(num_samples, len(like_files)))
        dislike_samples = random.sample(dislike_files, min(num_samples, len(dislike_files)))
        
        # Hiển thị mẫu "thích"
        plt.figure(figsize=(15, 5))
        for i, file in enumerate(like_samples):
            img = cv2.imread(str(file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(1, num_samples, i+1)
                plt.imshow(img)
                plt.title(f"Like {i+1}")
                plt.axis('off')
        plt.suptitle("Mẫu ảnh 'thích'", fontsize=16)
        plt.tight_layout()
        plt.savefig('like_samples.png')
        plt.show()
        
        # Hiển thị mẫu "không thích"
        plt.figure(figsize=(15, 5))
        for i, file in enumerate(dislike_samples):
            img = cv2.imread(str(file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(1, num_samples, i+1)
                plt.imshow(img)
                plt.title(f"Dislike {i+1}")
                plt.axis('off')
        plt.suptitle("Mẫu ảnh 'không thích'", fontsize=16)
        plt.tight_layout()
        plt.savefig('dislike_samples.png')
        plt.show()
        
        print("Các hình ảnh mẫu đã được lưu vào 'like_samples.png' và 'dislike_samples.png'")
        
    except Exception as e:
        print(f"Lỗi khi hiển thị mẫu dữ liệu: {e}")

def plot_training_history(history):
    """Vẽ biểu đồ lịch sử huấn luyện"""
    try:
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Lịch sử huấn luyện đã được lưu vào 'training_history.png'")
        
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ lịch sử huấn luyện: {e}")

if __name__ == "__main__":
    main()