import os
import time
import requests
import numpy as np
import json
import random
from datetime import datetime
from .model import PreferenceModel
from .image_processor import ImageProcessor
from .data_collector import DataCollector
from .tinder_client import TinderClient

class PreferenceRecognizer:
    """
    Lớp quản lý toàn bộ hệ thống nhận diện sở thích và tự động quẹt Tinder
    """
    
    def __init__(self, tinder_token=None, model_path=None, data_dir=None):
        """
        Khởi tạo hệ thống nhận diện sở thích
        
        Args:
            tinder_token: Token xác thực Tinder
            model_path: Đường dẫn đến model đã train
            data_dir: Thư mục lưu dữ liệu
        """
        # Khởi tạo các module
        self.model = PreferenceModel(model_path)
        self.image_processor = ImageProcessor(target_size=(224, 224))
        self.data_collector = DataCollector(data_dir)
        self.tinder_client = TinderClient(auth_token=tinder_token)
        
        # Thử tải model đã train
        self.model.load_trained_model()
        
        # Cấu hình
        self.config = {
            'threshold': 0.7,  # Ngưỡng xác suất để like (0.0 - 1.0)
            'max_profiles_per_day': 100,  # Số lượng profile tối đa xử lý mỗi ngày
            'delay_between_swipes': 2,  # Delay giữa các lần swipe (giây)
            'like_probability': 0.7,  # Xác suất like ngẫu nhiên khi không có model
            'collect_data': True  # Có thu thập dữ liệu mới không
        }
        
        # Metrics theo dõi hoạt động
        self.metrics = {
            'profiles_processed': 0,
            'likes': 0,
            'dislikes': 0,
            'matches': 0,
            'errors': 0,
            'session_start': datetime.now()
        }
        
        # Log file
        self.log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir, 
            f"swipe_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
    def log(self, message):
        """Ghi log ra file và console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
            
    def set_config(self, config_dict):
        """
        Cập nhật cấu hình
        
        Args:
            config_dict: Dictionary chứa các thông số cấu hình mới
        """
        self.config.update(config_dict)
        self.log(f"Đã cập nhật cấu hình: {json.dumps(config_dict)}")
        
    def train_model(self, epochs=20, batch_size=32):
        """
        Train model với dữ liệu hiện có
        
        Args:
            epochs: Số lượng epochs
            batch_size: Kích thước batch
            
        Returns:
            Kết quả training
        """
        # Kiểm tra có đủ dữ liệu để train không
        stats = self.data_collector.get_dataset_stats()
        
        if stats['total_count'] < 20:
            self.log(f"Không đủ dữ liệu để train model. Hiện có: {stats['total_count']} mẫu")
            return False
            
        if stats['like_count'] == 0 or stats['dislike_count'] == 0:
            self.log(f"Cần có cả dữ liệu like và dislike để train. Like: {stats['like_count']}, Dislike: {stats['dislike_count']}")
            return False
            
        self.log(f"Bắt đầu train model với {stats['total_count']} mẫu (Like: {stats['like_count']}, Dislike: {stats['dislike_count']})")
        
        # Train model
        history = self.model.train(
            training_dir=self.data_collector.data_dir,
            epochs=epochs,
            batch_size=batch_size
        )
        
        self.log(f"Train model hoàn tất. Accuracy: {history.history.get('accuracy', [-1])[-1]:.2f}")
        return history
        
    def predict_preference(self, image_url):
        """
        Dự đoán mức độ ưa thích với một ảnh
        
        Args:
            image_url: URL hoặc đường dẫn đến ảnh
            
        Returns:
            score: Điểm số ưa thích (0.0 - 1.0)
        """
        # Tải ảnh từ URL
        try:
            if image_url.startswith(('http://', 'https://')):
                response = requests.get(image_url)
                if response.status_code != 200:
                    return None
                    
                # Chuyển đổi sang numpy array
                image = np.asarray(bytearray(response.content), dtype="uint8")
                image = self.image_processor.preprocess_image(image)
            else:
                # Nếu là đường dẫn local
                image = self.image_processor.preprocess_image(image_url)
                
            # Trích xuất khuôn mặt từ ảnh (nếu có)
            faces = self.image_processor.extract_faces(image)
            
            if not faces:
                return None
                
            # Dự đoán cho mỗi khuôn mặt và lấy điểm cao nhất
            scores = []
            for face in faces:
                score = self.model.predict(face)
                scores.append(score)
                
            return max(scores) if scores else None
            
        except Exception as e:
            self.log(f"Lỗi khi dự đoán ảnh: {e}")
            return None
            
    def evaluate_profile(self, profile):
        """
        Đánh giá một profile và quyết định có nên like không
        
        Args:
            profile: Dữ liệu profile từ Tinder API
            
        Returns:
            decision: True nếu nên like, False nếu nên dislike
            score: Điểm số trung bình của profile
        """
        # Lấy các URL ảnh từ profile
        image_urls = self.tinder_client.extract_profile_images(profile)
        
        if not image_urls:
            self.log(f"Không tìm thấy ảnh trong profile {profile.get('_id', 'unknown')}")
            # Quyết định ngẫu nhiên nếu không có ảnh
            return random.random() < self.config['like_probability'], 0.5
            
        # Dự đoán mức độ ưa thích cho từng ảnh
        scores = []
        for url in image_urls:
            score = self.predict_preference(url)
            if score is not None:
                scores.append(score)
                
        # Nếu không có điểm nào hợp lệ, quyết định ngẫu nhiên
        if not scores:
            return random.random() < self.config['like_probability'], 0.5
            
        # Tính điểm trung bình
        avg_score = sum(scores) / len(scores)
        
        # Quyết định dựa trên ngưỡng
        decision = avg_score >= self.config['threshold']
        
        return decision, avg_score
        
    def auto_swipe(self, count=None, duration_minutes=None):
        """
        Tự động quẹt các profile
        
        Args:
            count: Số lượng profile xử lý (None = không giới hạn)
            duration_minutes: Thời gian chạy tính bằng phút (None = không giới hạn)
            
        Returns:
            stats: Kết quả thống kê
        """
        self.log(f"Bắt đầu tự động quẹt. Count: {count}, Duration: {duration_minutes} phút")
        
        # Reset metrics
        start_time = datetime.now()
        max_profiles = count or self.config['max_profiles_per_day']
        end_time = None
        
        if duration_minutes:
            import time
            from datetime import timedelta
            end_time = start_time + timedelta(minutes=duration_minutes)
            
        processed = 0
        
        while processed < max_profiles:
            # Kiểm tra thời gian
            if end_time and datetime.now() >= end_time:
                self.log(f"Đã hết thời gian ({duration_minutes} phút). Dừng quẹt.")
                break
                
            # Lấy recommendations
            recommendations = self.tinder_client.get_recs()
            
            if not recommendations:
                self.log("Không lấy được profile recommendations từ Tinder. Thử lại sau.")
                time.sleep(30)  # Đợi 30 giây
                continue
                
            # Xử lý từng profile
            for profile in recommendations:
                user_id = profile.get('_id')
                
                if not user_id:
                    continue
                    
                # Đánh giá profile
                like_decision, score = self.evaluate_profile(profile)
                
                # Thực hiện swipe
                if like_decision:
                    self.log(f"LIKE profile {user_id} với điểm số {score:.2f}")
                    match = self.tinder_client.like(user_id)
                    
                    if match:
                        self.metrics['matches'] += 1
                        self.log(f"MATCH với profile {user_id}!")
                        
                    self.metrics['likes'] += 1
                else:
                    self.log(f"PASS profile {user_id} với điểm số {score:.2f}")
                    self.tinder_client.dislike(user_id)
                    self.metrics['dislikes'] += 1
                    
                # Cập nhật số lượng đã xử lý
                processed += 1
                self.metrics['profiles_processed'] += 1
                
                # Thu thập dữ liệu nếu cấu hình cho phép
                if self.config['collect_data']:
                    try:
                        # Lấy ảnh đầu tiên
                        image_urls = self.tinder_client.extract_profile_images(profile)
                        if image_urls:
                            self.data_collector.save_image(
                                image_urls[0], 
                                is_like=like_decision,
                                source_id=user_id
                            )
                    except Exception as e:
                        self.log(f"Lỗi khi thu thập dữ liệu: {e}")
                        
                # Delay giữa các lần swipe
                time.sleep(self.config['delay_between_swipes'])
                
                # Kiểm tra điều kiện dừng
                if processed >= max_profiles:
                    break
                    
                if end_time and datetime.now() >= end_time:
                    break
                    
            # Đợi nếu không còn profile
            if not recommendations:
                time.sleep(60)  # Đợi 1 phút
                
        # Kết thúc
        duration = datetime.now() - start_time
        self.log(f"Hoàn thành tự động quẹt. Đã xử lý {processed} profiles trong {duration}")
        
        stats = {
            'duration': str(duration),
            'profiles_processed': processed,
            'likes': self.metrics['likes'],
            'dislikes': self.metrics['dislikes'],
            'matches': self.metrics['matches'],
            'like_rate': self.metrics['likes'] / processed if processed > 0 else 0
        }
        
        return stats