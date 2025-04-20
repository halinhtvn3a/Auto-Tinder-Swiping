import os
import sys
import io
import base64
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import module
root_dir = str(Path(__file__).parent.absolute())
sys.path.append(root_dir)

# Import các module cần thiết từ PreferenceRecognition
from PreferenceRecognition.model import PreferenceModel, EnsembleModel
from PreferenceRecognition.image_processor import ImageProcessor
from PreferenceRecognition.data_collector import DataCollector

app = Flask(__name__)
CORS(app)  # Cho phép CORS từ tất cả origins

# Khởi tạo các đối tượng
model = PreferenceModel()
ensemble_model = EnsembleModel()  # Khởi tạo thêm đối tượng EnsembleModel
image_processor = ImageProcessor()
data_collector = DataCollector()

# Biến để kiểm soát việc dùng ensemble hay model đơn lẻ
use_ensemble = True

# Thử tải model đã train
try:
    if use_ensemble:
        # Tải các model ensemble
        ensemble_loaded = ensemble_model._load_models()
        if ensemble_loaded:
            print(f"Đã tải thành công {len(ensemble_model.models)} mô hình ensemble!")
        else:
            # Nếu không tìm thấy model ensemble, thử tải model đơn lẻ
            print("Không tìm thấy mô hình ensemble. Thử tải mô hình đơn lẻ...")
            model_loaded = model.load_trained_model()
            if model_loaded:
                print("Đã tải thành công mô hình đơn lẻ đã huấn luyện!")
                use_ensemble = False
            else:
                print("Không tìm thấy mô hình đã huấn luyện. API sẽ hoạt động ở chế độ 'thu thập dữ liệu'.")
    else:
        model_loaded = model.load_trained_model()
        if model_loaded:
            print("Đã tải thành công mô hình đơn lẻ đã huấn luyện!")
        else:
            print("Không tìm thấy mô hình đã huấn luyện. API sẽ hoạt động ở chế độ 'thu thập dữ liệu'.")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    print("API sẽ hoạt động ở chế độ 'thu thập dữ liệu'.")
    use_ensemble = False

# Cấu hình
CONFIDENCE_THRESHOLD = 0.57  # Ngưỡng xác suất để xác định "thích"

@app.route('/')
def index():
    """Hiển thị trang web giao diện để test API"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_preference():
    """
    API endpoint nhận một ảnh và dự đoán xem người dùng thích hay không
    
    HTTP Method: POST
    Content-Type: multipart/form-data hoặc application/json
    Tham số:
        - image: File ảnh (form-data) hoặc base64 encoded string (JSON)
        - save: Boolean (tùy chọn) - Có lưu ảnh vào dataset không
        - feedback: Boolean (tùy chọn) - Phản hồi từ người dùng (true = thích, false = không thích)
        - use_ensemble: Boolean (tùy chọn) - Có sử dụng ensemble model không
    
    Kết quả:
        {
            "prediction": true/false,  # true = thích, false = không thích
            "confidence": 0.xx,        # độ tin cậy của dự đoán (0.0 - 1.0)
            "model_available": true/false,  # mô hình đã được tải hay không
            "ensemble_used": true/false,  # có sử dụng ensemble model không
            "individual_predictions": [...],  # [chỉ khi dùng ensemble] các dự đoán từ từng fold model 
            "image_saved": false,  # ảnh đã được lưu hay không
            "message": "..."           # thông báo bổ sung
        }
    """
    result = {
        "prediction": False,
        "confidence": 0.0,
        "model_available": model.model is not None or (hasattr(ensemble_model, 'models') and len(ensemble_model.models) > 0),
        "ensemble_used": use_ensemble and hasattr(ensemble_model, 'models') and len(ensemble_model.models) > 0,
        "individual_predictions": [],
        "image_saved": False,
        "message": ""
    }
    
    # Xác định có dùng ensemble hay không (từ tham số request hoặc giá trị mặc định)
    request_use_ensemble = use_ensemble
    if request.is_json and 'use_ensemble' in request.json:
        request_use_ensemble = bool(request.json['use_ensemble'])
    elif 'use_ensemble' in request.form:
        use_ensemble_value = request.form['use_ensemble'].lower()
        request_use_ensemble = use_ensemble_value in ('true', '1', 't', 'y', 'yes')
        
    # Cập nhật thông tin ensemble trong kết quả
    result["ensemble_used"] = request_use_ensemble and hasattr(ensemble_model, 'models') and len(ensemble_model.models) > 0
    
    try:
        image_data = None
        
        # Xử lý dữ liệu đầu vào từ form-data
        if 'image' in request.files:
            image_file = request.files['image']
            image_data = Image.open(io.BytesIO(image_file.read()))
            
        # Xử lý dữ liệu đầu vào từ JSON (base64)
        elif request.is_json and 'image' in request.json:
            base64_data = request.json['image']
            if ',' in base64_data:  # Định dạng: data:image/jpeg;base64,/9j/...
                base64_data = base64_data.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
            image_data = Image.open(io.BytesIO(image_bytes))
            
        else:
            return jsonify({
                "error": "Không tìm thấy ảnh. Vui lòng gửi ảnh qua form-data (key='image') hoặc JSON với base64 string (key='image')"
            }), 400
        
        # Chuyển đổi sang numpy array và xử lý ảnh
        img_array = np.array(image_data)
        processed_img = image_processor.preprocess_image(img_array)
        
        # Dự đoán dựa trên model có sẵn
        if result["ensemble_used"]:
            # Sử dụng ensemble model để dự đoán
            confidence, individual_preds = ensemble_model.predict(processed_img)
            prediction = confidence >= CONFIDENCE_THRESHOLD
            
            result["prediction"] = bool(prediction)
            result["confidence"] = float(confidence)
            result["individual_predictions"] = [float(pred) for pred in individual_preds]
            result["message"] = f"[Ensemble] Dự đoán {'THÍCH' if prediction else 'KHÔNG THÍCH'} với độ tin cậy {confidence:.2f}"
            
        elif model.model is not None:
            # Sử dụng model đơn lẻ để dự đoán
            confidence = model.predict(processed_img)
            prediction = confidence >= CONFIDENCE_THRESHOLD
            
            result["prediction"] = bool(prediction)
            result["confidence"] = float(confidence)
            result["message"] = f"Dự đoán {'THÍCH' if prediction else 'KHÔNG THÍCH'} với độ tin cậy {confidence:.2f}"
        
        else:
            result["message"] = "Model chưa được huấn luyện. Không thể dự đoán."
        
        # Lưu ảnh nếu được yêu cầu
        should_save = False
        is_like = False
        
        # Lấy thông tin phản hồi của người dùng (nếu có)
        if request.is_json and 'feedback' in request.json:
            should_save = True
            is_like = bool(request.json['feedback'])
        elif request.form and 'feedback' in request.form:
            should_save = True
            feedback_value = request.form['feedback'].lower()
            is_like = feedback_value in ('true', '1', 't', 'y', 'yes')
            
        # Hoặc lưu theo tham số save và kết quả dự đoán
        elif request.is_json and 'save' in request.json and request.json['save']:
            should_save = True
            is_like = result["prediction"] if result["model_available"] else False
        elif request.form and 'save' in request.form:
            save_value = request.form['save'].lower()
            should_save = save_value in ('true', '1', 't', 'y', 'yes')
            is_like = result["prediction"] if result["model_available"] else False
            
        # Lưu ảnh nếu được yêu cầu
        if should_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = data_collector.save_image(processed_img, is_like, source_id=f"api_{timestamp}")
            result["image_saved"] = True
            result["saved_path"] = filepath
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": f"Lỗi khi xử lý ảnh: {str(e)}"
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    API endpoint để huấn luyện mô hình với dữ liệu hiện có
    
    HTTP Method: POST
    Content-Type: application/json
    Tham số:
        - epochs: số lượng epochs (mặc định: 20)
        - batch_size: kích thước batch (mặc định: 32)
        
    Kết quả:
        {
            "success": true/false,
            "message": "...",
            "stats": {
                "like_count": x,
                "dislike_count": y,
                "total_count": z,
                "is_balanced": true/false
            }
        }
    """
    # Lấy thông số huấn luyện
    epochs = 20
    batch_size = 32
    
    if request.is_json:
        if 'epochs' in request.json:
            epochs = int(request.json['epochs'])
        if 'batch_size' in request.json:
            batch_size = int(request.json['batch_size'])
            
    # Kiểm tra dữ liệu huấn luyện
    stats = data_collector.get_dataset_stats()
    
    # Kiểm tra có đủ dữ liệu không
    if stats['like_count'] < 10 or stats['dislike_count'] < 10:
        return jsonify({
            "success": False,
            "message": f"Không đủ dữ liệu để huấn luyện. Cần ít nhất 10 mẫu mỗi loại. Hiện có: Like={stats['like_count']}, Dislike={stats['dislike_count']}",
            "stats": stats
        })
        
    try:
        # Khởi tạo model
        model.build_model(transfer_learning=True)
        
        # Huấn luyện model
        history = model.train(
            training_dir=data_collector.data_dir,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Lấy accuracy cuối cùng
        final_accuracy = float(history.history.get('accuracy', [-1])[-1])
        
        return jsonify({
            "success": True,
            "message": f"Đã huấn luyện mô hình thành công với accuracy {final_accuracy:.4f}",
            "stats": stats,
            "accuracy": final_accuracy
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Lỗi khi huấn luyện mô hình: {str(e)}",
            "stats": stats
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    API endpoint để lấy thông tin về dataset
    
    HTTP Method: GET
    
    Kết quả:
        {
            "like_count": x,
            "dislike_count": y,
            "total_count": z,
            "is_balanced": true/false,
            "model_available": true/false
        }
    """
    stats = data_collector.get_dataset_stats()
    stats["model_available"] = model.model is not None
    
    return jsonify(stats)

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint kiểm tra trạng thái hoạt động"""
    return jsonify({
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "model_available": model.model is not None,
        "dataset_stats": data_collector.get_dataset_stats()
    })

if __name__ == '__main__':
    # Cổng mặc định: 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Debug mode cho phát triển
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Starting API server on port {port}...")
    print(f"Model loaded: {model.model is not None}")
    print(f"Dataset stats: {data_collector.get_dataset_stats()}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)