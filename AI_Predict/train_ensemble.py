"""
Script để huấn luyện mô hình nhận diện sở thích sử dụng kỹ thuật Ensemble 5-Fold Cross-Validation với PyTorch

Usage:
    python train_ensemble.py [--epochs 30] [--batch-size 16] [--base-network MobileNetV2]
    python AI_Predict\train_ensemble.py --epochs 20 --batch-size 32 --base-network ResNet50
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import threading
import subprocess

# === CẤU HÌNH GPU CHO PYTORCH ===
# Thiết lập các biến môi trường 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_USE_CUDA_DSA"] = "0"  # Disable CUDA Dynamic Shapes Analysis
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["DISABLE_PYTORCH_ONEDNN_OPTIMIZATIONS"] = "1"  # Disable oneDNN optimizations
os.environ["MKL_VERBOSE"] = "0"  # Turns off MKL verbose messages

print("===== KIỂM TRA CUDA VỚI PYTORCH =====")
print(f"Python version: {sys.version}")

# Kiểm tra CUDA với PyTorch
try:
    import torch
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "không xác định"
        
        print(f"✅ CUDA available: {cuda_available}")
        print(f"✅ Số lượng GPU: {gpu_count}")
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ CUDA version: {cuda_version}")
        
        # Kiểm tra hiệu năng GPU
        print("\nĐang kiểm tra hiệu năng GPU...")
        
        # Tạo các tensor lớn để kiểm tra
        size = 5000
        
        # Làm nóng GPU
        torch.cuda.synchronize()
        x_gpu = torch.randn(size, size, device='cuda')
        y_gpu = torch.randn(size, size, device='cuda')
        torch.cuda.synchronize()
        
        # Benchmark CPU
        start_time = time.time()
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        # Benchmark GPU
        torch.cuda.synchronize()
        start_time = time.time()
        z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        # So sánh hiệu năng
        speedup = cpu_time / gpu_time
        print(f"Thời gian CPU: {cpu_time:.4f} giây")
        print(f"Thời gian GPU: {gpu_time:.4f} giây")
        print(f"Tăng tốc: {speedup:.2f}x")
        
        if speedup > 5:
            print("✅ GPU hoạt động hiệu quả!")
        elif speedup > 1:
            print("✅ GPU hoạt động tốt")
        else:
            print("⚠️ GPU không nhanh hơn CPU!")
    else:
        print("❌ CUDA không khả dụng")
        print("Đang chuyển sang chạy trên CPU...")
except ImportError:
    print("PyTorch không được cài đặt, không thể kiểm tra CUDA")
    print("Để cài đặt PyTorch: pip install torch torchvision torchaudio")
    cuda_available = False

# Import các module PyTorch cần thiết - Import trực tiếp từ pytorch_model
try:
    # Import trực tiếp để tránh phụ thuộc vào __init__.py
    from PreferenceRecognition.model import EnsembleModel
    from PreferenceRecognition.data_collector import DataCollector
    print("✅ Đã import thành công các module PyTorch")
except ImportError as e:
    print(f"❌ Lỗi khi import modules PyTorch: {e}")
    print("\nLỗi này có thể do:")
    print("1. Module PyTorch chưa được cài đặt đúng cách")
    print("2. Đường dẫn tới thư mục PreferenceRecognition không đúng")
    sys.exit(1)

# Biến theo dõi hiệu suất GPU
gpu_usage_data = []
gpu_timestamps = []
monitoring_active = False

def monitor_gpu_usage():
    """Theo dõi sử dụng GPU trong quá trình huấn luyện"""
    global gpu_usage_data, gpu_timestamps, monitoring_active
    try:
        import subprocess
        monitoring_active = True
        start_time = time.time()
        
        while monitoring_active:
            try:
                output = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    universal_newlines=True
                ).strip()
                
                if ',' in output:
                    gpu_util, mem_used = output.split(',')
                    usage = float(gpu_util)
                    memory = float(mem_used)
                    
                    gpu_usage_data.append((usage, memory))
                    gpu_timestamps.append(time.time() - start_time)
                    
                    # Hiển thị thông tin GPU mỗi 30 giây
                    # if len(gpu_timestamps) % 15 == 0:
                    #     print(f"GPU: {usage:.1f}% | Memory: {memory:.0f} MiB | Elapsed: {gpu_timestamps[-1]:.1f}s")
            except:
                pass
                
            time.sleep(2)
    except:
        pass

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình Ensemble với 5-Fold Cross-Validation (PyTorch)')
    parser.add_argument('--epochs', type=int, default=20, help='Số epochs để huấn luyện')
    parser.add_argument('--batch-size', type=int, default=32, help='Kích thước batch')
    parser.add_argument('--base-network', type=str, default='ResNet50', 
                      choices=['MobileNetV2', 'ResNet50', 'VGG16', 'EfficientNet'], 
                      help='Mạng cơ sở cho transfer learning')
    parser.add_argument('--n-folds', type=int, default=5, help='Số lượng fold cho cross-validation')
    parser.add_argument('--no-transfer-learning', action='store_true',
                      help='Không sử dụng transfer learning (xây dựng model từ đầu)')
    parser.add_argument('--no-monitor-gpu', action='store_true',
                      help='Không theo dõi hiệu suất GPU trong quá trình huấn luyện')
    args = parser.parse_args()
    
    # Bắt đầu theo dõi GPU nếu có
    gpu_monitor_thread = None
    if cuda_available and not args.no_monitor_gpu:
        try:
            gpu_monitor_thread = threading.Thread(target=monitor_gpu_usage)
            gpu_monitor_thread.daemon = True
            gpu_monitor_thread.start()
            print("✅ Đã bắt đầu theo dõi hiệu suất GPU")
        except:
            print("❌ Không thể theo dõi hiệu suất GPU")
    
    # Khởi tạo model và data collector
    ensemble_model = EnsembleModel(n_folds=args.n_folds)
    data_collector = DataCollector()
    
    # Xác định đường dẫn data
    data_dir = data_collector.data_dir
    
    # Kiểm tra data
    stats = data_collector.get_dataset_stats()
    print(f"\nThống kê dataset:")
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

    if not stats['is_balanced']:
        print("\nDỮ LIỆU KHÔNG CÂN BẰNG: Tỉ lệ lớp 'like' và 'dislike' chênh lệch lớn.")
        print("Khuyến nghị: Chạy 'python balance_dataset.py' trước khi huấn luyện để cân bằng dữ liệu")
        print("Hoặc --ratio=0.5 để tạo tỉ lệ 1:2 thay vì 1:1 (dễ huấn luyện hơn)")
        
        proceed = input("Bạn có muốn tiếp tục mà không cần cân bằng dữ liệu? (y/n): ")
        if proceed.lower() != 'y':
            print("Hủy quá trình huấn luyện.")
            return
    
    print(f"\nBắt đầu huấn luyện mô hình Ensemble {args.n_folds}-Fold Cross-Validation (PyTorch)")
    print(f"- Epochs mỗi fold: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Transfer learning: {'Không' if args.no_transfer_learning else 'Có'}")
    print(f"- Mạng cơ sở: {args.base_network}")
    print(f"- Sử dụng GPU: {'Có' if cuda_available else 'Không'}")
    
    # Huấn luyện mô hình ensemble
    try:
        start_time = time.time()
        print("\nBắt đầu quá trình huấn luyện với PyTorch...")
        
        # Huấn luyện mô hình ensemble với PyTorch
        histories = ensemble_model.train(
            data_dir=data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_network=args.base_network,
            transfer_learning=not args.no_transfer_learning
        )
        
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nHuấn luyện hoàn tất! Thời gian: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Vẽ biểu đồ accuracy và loss cho tất cả các fold
        plot_ensemble_training_history(histories)
        
        # Vẽ biểu đồ sử dụng GPU nếu có dữ liệu
        if gpu_monitor_thread:
            global monitoring_active
            monitoring_active = False
            gpu_monitor_thread.join(timeout=1)
            plot_gpu_usage()
        
        # Đánh giá mô hình
        evaluate_ensemble_model(ensemble_model, data_dir)
        
    except Exception as e:
        print(f"Lỗi khi huấn luyện mô hình ensemble: {e}")
        import traceback
        traceback.print_exc()

def plot_gpu_usage():
    """Vẽ biểu đồ sử dụng GPU sau khi huấn luyện"""
    global gpu_usage_data, gpu_timestamps
    
    if not gpu_usage_data:
        return
    
    try:
        plt.figure(figsize=(14, 6))
        
        # Tách dữ liệu thành hai mảng
        gpu_util = [x[0] for x in gpu_usage_data]
        gpu_mem = [x[1] for x in gpu_usage_data]
        
        # Vẽ biểu đồ sử dụng GPU
        plt.subplot(1, 2, 1)
        plt.plot(gpu_timestamps, gpu_util, label='GPU Utilization %', color='green')
        plt.title('Hiệu suất GPU trong quá trình huấn luyện', fontsize=14)
        plt.xlabel('Thời gian (giây)', fontsize=12)
        plt.ylabel('GPU Utilization (%)', fontsize=12)
        plt.ylim(0, 105)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Vẽ biểu đồ sử dụng bộ nhớ GPU
        plt.subplot(1, 2, 2)
        plt.plot(gpu_timestamps, gpu_mem, label='GPU Memory (MiB)', color='blue')
        plt.title('Bộ nhớ GPU trong quá trình huấn luyện', fontsize=14)
        plt.xlabel('Thời gian (giây)', fontsize=12)
        plt.ylabel('GPU Memory (MiB)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Tính thống kê
        avg_util = np.mean(gpu_util)
        max_util = np.max(gpu_util)
        avg_mem = np.mean(gpu_mem)
        max_mem = np.max(gpu_mem)
        
        plt.figtext(0.5, 0.01, 
                f"Trung bình: {avg_util:.1f}% | Tối đa: {max_util:.1f}% | Bộ nhớ TB: {avg_mem:.0f} MiB | Bộ nhớ tối đa: {max_mem:.0f} MiB", 
                ha='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('gpu_usage_history.png', dpi=300)
        print("Đã lưu biểu đồ sử dụng GPU vào 'gpu_usage_history.png'")
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ GPU: {e}")

def plot_ensemble_training_history(histories):
    """Vẽ biểu đồ lịch sử huấn luyện cho tất cả các fold"""
    try:
        # Số lượng fold
        n_folds = len(histories)
        
        # Tạo figure
        plt.figure(figsize=(15, 10))
        
        # Plot accuracy
        plt.subplot(2, 1, 1)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_folds))
        
        # Vẽ đường accuracy cho từng fold
        for i, history in enumerate(histories):
            plt.plot(history['accuracy'], label=f'Fold {i+1} Train', 
                    color=colors[i], alpha=0.7, linestyle='-')
            plt.plot(history['val_accuracy'], label=f'Fold {i+1} Val', 
                    color=colors[i], alpha=0.7, linestyle='--')
            
        plt.title('Accuracy qua các fold', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Plot loss
        plt.subplot(2, 1, 2)
        
        # Vẽ đường loss cho từng fold
        for i, history in enumerate(histories):
            plt.plot(history['loss'], label=f'Fold {i+1} Train', 
                    color=colors[i], alpha=0.7, linestyle='-')
            plt.plot(history['val_loss'], label=f'Fold {i+1} Val', 
                    color=colors[i], alpha=0.7, linestyle='--')
            
        plt.title('Loss qua các fold', fontsize=14)
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ensemble_training_history.png', dpi=300)
        print("Đã lưu biểu đồ huấn luyện vào 'ensemble_training_history.png'")
        
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ: {e}")

def evaluate_ensemble_model(ensemble_model, test_dir):
    """Đánh giá mô hình PyTorch ensemble"""
    from pathlib import Path
    import cv2
    from PIL import Image
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
    
    try:
        print("\nĐánh giá mô hình ensemble trên dữ liệu...")
        
        # Thu thập đường dẫn hình ảnh và nhãn
        like_files = list(Path(os.path.join(test_dir, 'like')).glob('*.jpg'))
        dislike_files = list(Path(os.path.join(test_dir, 'dislike')).glob('*.jpg'))
        
        # Nếu không có thư mục test riêng, dùng tất cả dữ liệu
        if len(like_files) == 0 or len(dislike_files) == 0:
            # Sử dụng K-fold để đánh giá
            print("Không tìm thấy thư mục test riêng. Sử dụng toàn bộ dữ liệu để đánh giá...")
            
            # Thu thập đường dẫn và nhãn
            all_files = []
            all_labels = []
            
            for file_path in Path(os.path.join(test_dir, 'like')).glob('*.jpg'):
                all_files.append(str(file_path))
                all_labels.append(1)
                
            for file_path in Path(os.path.join(test_dir, 'dislike')).glob('*.jpg'):
                all_files.append(str(file_path))
                all_labels.append(0)
        else:
            # Sử dụng tập test có sẵn
            all_files = [str(f) for f in like_files] + [str(f) for f in dislike_files]
            all_labels = [1] * len(like_files) + [0] * len(dislike_files)
            
        # Dự đoán trên tất cả hình ảnh
        y_true = []
        y_pred = []
        y_scores = []
        
        for file_path, label in zip(all_files, all_labels):
            # Đọc và tiền xử lý hình ảnh
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Dự đoán
            prob = ensemble_model.predict(img)
            pred = 1 if prob >= 0.5 else 0
            
            y_true.append(label)
            y_pred.append(pred)
            y_scores.append(prob)
        
        # Tính các chỉ số đánh giá
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        # Báo cáo phân loại
        report = classification_report(y_true, y_pred, target_names=['Dislike', 'Like'], output_dict=True)
        
        # Tạo kết quả
        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_scores
        }
        
        # In kết quả
        print(f"Độ chính xác của mô hình ensemble: {results['accuracy']:.4f}")
        
        # Vẽ confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Ensemble Model - Ma trận nhầm lẫn', fontsize=15)
        plt.colorbar()
        plt.xticks([0, 1], ['Dislike', 'Like'], fontsize=12)
        plt.yticks([0, 1], ['Dislike', 'Like'], fontsize=12)
        plt.xlabel('Dự đoán', fontsize=12)
        plt.ylabel('Thực tế', fontsize=12)
        
        # Hiển thị số lượng trong từng ô
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        horizontalalignment="center",
                        fontsize=14, fontweight='bold',
                        color="white" if cm[i, j] > cm.max() / 2. else "black")
        
        plt.tight_layout()
        plt.savefig('ensemble_confusion_matrix.png', dpi=300)
        print("Đã lưu ma trận nhầm lẫn vào 'ensemble_confusion_matrix.png'")
        
        # Vẽ ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('ensemble_roc_curve.png', dpi=300)
        print("Đã lưu ROC curve vào 'ensemble_roc_curve.png'")
        
        # Hiển thị báo cáo phân loại
        print("\nBáo cáo phân loại chi tiết:")
        print(f"                 Precision    Recall  F1-Score   Support")
        print(f"Dislike         {report['Dislike']['precision']:.4f}     {report['Dislike']['recall']:.4f}    {report['Dislike']['f1-score']:.4f}       {report['Dislike']['support']}")
        print(f"Like            {report['Like']['precision']:.4f}     {report['Like']['recall']:.4f}    {report['Like']['f1-score']:.4f}       {report['Like']['support']}")
        print(f"Accuracy                          {report['accuracy']:.4f}")
        print(f"Macro Avg       {report['macro avg']['precision']:.4f}     {report['macro avg']['recall']:.4f}    {report['macro avg']['f1-score']:.4f}       {report['macro avg']['support']}")
        print(f"Weighted Avg    {report['weighted avg']['precision']:.4f}     {report['weighted avg']['recall']:.4f}    {report['weighted avg']['f1-score']:.4f}       {report['weighted avg']['support']}")
        
        # Hiển thị một số dự đoán sai
        incorrect_predictions = [(file, true, pred, score) 
                                for file, true, pred, score in zip(all_files, y_true, y_pred, y_scores)
                                if true != pred]
        
        if incorrect_predictions:
            # Hiển thị tối đa 9 hình ảnh bị dự đoán sai
            n_images = min(9, len(incorrect_predictions))
            
            plt.figure(figsize=(15, 12))
            for i in range(n_images):
                file, true, pred, score = incorrect_predictions[i]
                img = cv2.imread(file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.subplot(3, 3, i+1)
                plt.imshow(img)
                plt.title(f"True: {'Like' if true == 1 else 'Dislike'}\nPred: {'Like' if pred == 1 else 'Dislike'}\nScore: {score:.2f}")
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('ensemble_incorrect_predictions.png', dpi=300)
            print("Đã lưu các dự đoán sai vào 'ensemble_incorrect_predictions.png'")
        
        return results
        
    except Exception as e:
        print(f"Lỗi khi đánh giá mô hình ensemble: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()