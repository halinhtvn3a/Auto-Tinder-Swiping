"""
Script để đánh giá mô hình nhận diện sở thích và phân tích lỗi

Usage:
    python evaluate_model.py [--data-dir PATH_TO_DATA] [--ensemble]
    python evaluate_model.py --ensemble
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import cv2
from PreferenceRecognition.model import PreferenceModel, EnsembleModel
from PreferenceRecognition.data_collector import DataCollector
from PreferenceRecognition.image_processor import ImageProcessor
import tensorflow as tf
import glob
import torch

def main():
    parser = argparse.ArgumentParser(description='Đánh giá mô hình nhận diện sở thích')
    parser.add_argument('--data-dir', type=str, default=None, 
                      help='Đường dẫn đến thư mục chứa dữ liệu (nếu khác mặc định)')
    parser.add_argument('--ensemble', action='store_true',
                      help='Sử dụng ensemble model thay vì model đơn')
    args = parser.parse_args()
    
    # Khởi tạo các đối tượng
    data_collector = DataCollector(data_dir=args.data_dir)
    image_processor = ImageProcessor()
    
    # Lấy đường dẫn dữ liệu
    data_dir = data_collector.data_dir
    print(f"Đánh giá model với dữ liệu từ: {data_dir}")
    
    # Tạo generator cho tập test (chúng ta dùng toàn bộ dữ liệu vì dataset còn nhỏ)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),  # Standard image size used in your models
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    
    if args.ensemble:
        evaluate_ensemble(test_generator, data_dir)
    else:
        evaluate_single_model(test_generator, data_dir)
    
def evaluate_single_model(test_generator, data_dir):
    """Đánh giá model đơn lẻ"""
    model = PreferenceModel()
    
    # Tải model
    if not model.load_trained_model():
        print("Không tìm thấy model đã train. Vui lòng huấn luyện model trước.")
        return
    
    # Đánh giá tổng quát
    print("\nĐánh giá tổng quát:")
    scores = model.model.evaluate(test_generator)
    print(f"Loss: {scores[0]:.4f}")
    print(f"Accuracy: {scores[1]:.4f}")
    
    # Reset lại generator để dùng cho predict
    test_generator.reset()
    
    # Dự đoán
    predictions = model.model.predict(test_generator)
    predicted_classes = (predictions > 0.5).astype(int)
    true_classes = test_generator.classes
    
    # Hiển thị báo cáo phân loại chi tiết
    display_evaluation_results(true_classes, predicted_classes, predictions, test_generator=test_generator)

def evaluate_ensemble(test_generator, data_dir):
    """Đánh giá ensemble model sử dụng PyTorch"""
    
    # Tìm tất cả các mô hình ensemble (5 fold)
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PreferenceRecognition', 'saved_models', 'ensemble')
    model_paths = glob.glob(os.path.join(model_dir, "preference_model_fold*.h5"))
    
    if not model_paths:
        print(f"Không tìm thấy model ensemble nào trong {model_dir}. Vui lòng huấn luyện ensemble model trước.")
        return
    
    print(f"Đã tìm thấy {len(model_paths)} mô hình trong ensemble.")
    
    # Tải các model
    models = []
    for path in model_paths:
        fold_model = PreferenceModel(model_path=path)
        # Khởi tạo kiến trúc mô hình trước khi tải trọng số
        fold_model.build_model()  # Phải gọi build_model() trước load_trained_model()
        if fold_model.load_trained_model():
            models.append(fold_model)
            print(f"Đã tải model: {os.path.basename(path)}")
    
    if not models:
        print("Không thể tải bất kỳ model nào. Kiểm tra lại các file model.")
        return
    
    # Reset generator trước khi dự đoán
    test_generator.reset()
    
    # Dự đoán từ tất cả các model trong ensemble
    all_predictions = []
    
    print("\nĐang thực hiện dự đoán với tất cả các model...")
    
    # Dự đoán cho từng model
    for i, model in enumerate(models):
        print(f"Đang dự đoán với model {i+1}/{len(models)}...")
        predictions = []
        
        # PyTorch cần một cách khác để dự đoán
        batch_idx = 0
        for inputs, _ in test_generator:
            # Chuyển đổi numpy array thành tensor PyTorch và permute channels
            inputs_tensor = torch.from_numpy(inputs).permute(0, 3, 1, 2).float().to(model.device)
            
            # Dự đoán với PyTorch
            with torch.no_grad():
                outputs = model.model(inputs_tensor)
                predictions.extend(outputs.cpu().numpy().flatten())
            
            # Kiểm tra nếu đã qua hết tất cả các batch
            batch_idx += 1
            if batch_idx * test_generator.batch_size >= test_generator.n:
                break
                
        # Cắt predictions nếu cần thiết
        predictions = predictions[:test_generator.n]
        all_predictions.append(predictions)
        test_generator.reset()  # Reset generator sau mỗi lần dự đoán
    
    # Tổng hợp các dự đoán bằng cách lấy trung bình
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    # Chuyển thành nhãn dựa trên ngưỡng
    predicted_classes = (ensemble_predictions > 0.5).astype(int)
    true_classes = test_generator.classes
    
    print("\nKết quả đánh giá Ensemble model:")
    # Hiển thị báo cáo phân loại chi tiết
    display_evaluation_results(true_classes, predicted_classes, ensemble_predictions, is_ensemble=True, test_generator=test_generator)

def display_evaluation_results(true_classes, predicted_classes, predictions, is_ensemble=False, test_generator=None):
    """Hiển thị kết quả đánh giá và vẽ các biểu đồ"""
    prefix = "ensemble_" if is_ensemble else ""
    
    # Hiển thị báo cáo phân loại chi tiết
    print("\nBáo cáo phân loại chi tiết:")
    print(classification_report(true_classes, predicted_classes, target_names=['Dislike', 'Like']))
    
    # Ma trận nhầm lẫn
    print("\nMa trận nhầm lẫn:")
    cm = confusion_matrix(true_classes, predicted_classes)
    print(cm)
    
    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ma trận nhầm lẫn' + (" (Ensemble)" if is_ensemble else ""))
    plt.colorbar()
    plt.xticks([0, 1], ['Dislike', 'Like'])
    plt.yticks([0, 1], ['Dislike', 'Like'])
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    
    # Hiển thị số lượng trong từng ô
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    
    plt.tight_layout()
    plt.savefig(f'{prefix}confusion_matrix.png')
    print(f"Đã lưu ma trận nhầm lẫn vào file: {prefix}confusion_matrix.png")
    
    # Vẽ đường cong ROC
    fpr, tpr, _ = roc_curve(true_classes, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic' + (" (Ensemble)" if is_ensemble else ""))
    plt.legend(loc="lower right")
    plt.savefig(f'{prefix}roc_curve.png')
    print(f"Đã lưu đường cong ROC vào file: {prefix}roc_curve.png")
    
    # Phân tích các dự đoán sai
    if test_generator is not None:
        analyze_incorrect_predictions(predictions, true_classes, test_generator, prefix)
    
def analyze_incorrect_predictions(predictions, true_classes, test_generator, prefix="", num_examples=10):
    """Phân tích và hiển thị các ví dụ về dự đoán sai"""
    # Reset generator
    test_generator.reset()
    
    pred_classes = (predictions > 0.5).astype(int).flatten()
    incorrect_indices = np.where(pred_classes != true_classes)[0]
    
    if len(incorrect_indices) == 0:
        print("\nKhông có dự đoán sai trong tập dữ liệu! (Điều này hiếm khi xảy ra)")
        return
        
    print(f"\nTổng số dự đoán sai: {len(incorrect_indices)}/{len(true_classes)} ({len(incorrect_indices)/len(true_classes)*100:.1f}%)")
    
    # Lấy số lượng mẫu cần hiển thị (hoặc tất cả nếu ít hơn)
    n_examples = min(num_examples, len(incorrect_indices))
    sample_indices = np.random.choice(incorrect_indices, n_examples, replace=False)
    
    # Gom các ảnh đã dự đoán sai
    incorrect_images = []
    incorrect_labels = []
    incorrect_preds = []
    incorrect_confs = []
    
    # Lấy đường dẫn tệp từ generator
    file_paths = test_generator.filepaths
    
    for idx in sample_indices:
        img_path = file_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        true_label = "Like" if true_classes[idx] == 1 else "Dislike"
        pred_label = "Like" if pred_classes[idx] == 1 else "Dislike"
        
        # Lấy ra giá trị confidence (probability)
        if predictions.ndim > 1:
            confidence = predictions[idx][0]
        else:
            confidence = predictions[idx]
        
        incorrect_images.append(img)
        incorrect_labels.append(true_label)
        incorrect_preds.append(pred_label)
        incorrect_confs.append(confidence)
    
    # Vẽ các ảnh dự đoán sai
    plt.figure(figsize=(20, 4 * ((n_examples + 4) // 5)))
    
    for i in range(n_examples):
        plt.subplot((n_examples + 4) // 5, 5, i + 1)
        plt.imshow(incorrect_images[i])
        plt.title(f"True: {incorrect_labels[i]}\nPred: {incorrect_preds[i]}\nConf: {incorrect_confs[i]:.2f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{prefix}incorrect_predictions.png')
    print(f"Đã lưu hình ảnh dự đoán sai vào file: {prefix}incorrect_predictions.png")
    
    print("\nGợi ý cải thiện dựa trên các dự đoán sai:")
    print("1. Kiểm tra các ảnh dự đoán sai để tìm pattern chung")
    print("2. Thêm dữ liệu tương tự với các trường hợp bị phân loại sai")
    print("3. Xem xét điều chỉnh ngưỡng dự đoán (hiện tại là 0.5) nếu có xu hướng lệch")
    
def show_activation_maps(model, img_path):
    """Hiển thị activation maps của mô hình trên một ảnh cụ thể"""
    # Tính năng nâng cao, sẽ được triển khai sau
    pass
    
if __name__ == "__main__":
    main()