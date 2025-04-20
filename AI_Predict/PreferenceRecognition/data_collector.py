import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen, Request
import shutil

class DataCollector:
    """
    Class xử lý việc thu thập và quản lý dữ liệu cho model nhận diện sở thích
    """
    
    def __init__(self, data_dir=None):
        """
        Khởi tạo DataCollector
        
        Args:
            data_dir: Thư mục lưu dữ liệu. Mặc định là ./data
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'data')
        self.like_dir = os.path.join(self.data_dir, 'like')
        self.dislike_dir = os.path.join(self.data_dir, 'dislike')
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.like_dir, exist_ok=True)
        os.makedirs(self.dislike_dir, exist_ok=True)
        
    def save_image(self, image, is_like, source_id=None, url=None):
        """
        Lưu ảnh vào thư mục phù hợp (like hoặc dislike)
        
        Args:
            image: Numpy array hoặc đường dẫn đến ảnh
            is_like: True nếu là ảnh "thích", False nếu "không thích"
            source_id: ID của profile (nếu có)
            url: URL của ảnh (nếu có)
            
        Returns:
            path: Đường dẫn đến ảnh đã lưu
        """
        # Xác định thư mục đích
        target_dir = self.like_dir if is_like else self.dislike_dir
        
        # Tạo tên file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_suffix = f"_{source_id}" if source_id else ""
        filename = f"{timestamp}{source_suffix}.jpg"
        filepath = os.path.join(target_dir, filename)
        
        # Lưu ảnh
        if isinstance(image, str):
            # Nếu là đường dẫn, copy file
            if os.path.exists(image):
                shutil.copy2(image, filepath)
            elif image.startswith(('http://', 'https://')):
                # Nếu là URL, download ảnh
                self._download_image(image, filepath)
            else:
                raise ValueError("Đường dẫn ảnh không hợp lệ")
        else:
            # Nếu là numpy array, lưu trực tiếp
            cv2.imwrite(filepath, image)
            
        return filepath
        
    def _download_image(self, url, save_path):
        """Tải ảnh từ URL và lưu vào đường dẫn"""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        try:
            req = Request(url=url, headers=headers)
            with urlopen(req) as response, open(save_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            return True
        except Exception as e:
            print(f"Lỗi khi tải ảnh: {e}")
            return False
            
    def load_dataset(self):
        """
        Tải toàn bộ dataset từ thư mục
        
        Returns:
            X: numpy array của các ảnh
            y: numpy array của labels (1: like, 0: dislike)
        """
        like_files = list(Path(self.like_dir).glob('*.jpg'))
        dislike_files = list(Path(self.dislike_dir).glob('*.jpg'))
        
        if not like_files and not dislike_files:
            print("Không tìm thấy dữ liệu trong thư mục")
            return None, None
            
        # Đọc ảnh và tạo labels
        images = []
        labels = []
        
        # Đọc ảnh "thích"
        for file in like_files:
            img = cv2.imread(str(file))
            if img is not None:
                images.append(img)
                labels.append(1)
                
        # Đọc ảnh "không thích"
        for file in dislike_files:
            img = cv2.imread(str(file))
            if img is not None:
                images.append(img)
                labels.append(0)
                
        # Convert thành numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        return X, y
        
    def get_dataset_stats(self):
        """
        Lấy thông tin về dataset hiện tại
        
        Returns:
            stats: Dictionary chứa thông tin về dataset
        """
        like_files = list(Path(self.like_dir).glob('*.jpg'))
        dislike_files = list(Path(self.dislike_dir).glob('*.jpg'))
        
        stats = {
            'like_count': len(like_files),
            'dislike_count': len(dislike_files),
            'total_count': len(like_files) + len(dislike_files),
            'is_balanced': abs(len(like_files) - len(dislike_files)) <= 10
        }
        
        return stats