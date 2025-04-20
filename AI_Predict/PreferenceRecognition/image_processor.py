import cv2
import numpy as np
from PIL import Image
import io
import base64

class ImageProcessor:
    """
    Class xử lý ảnh để chuẩn bị cho model nhận diện
    """
    
    def __init__(self, target_size=(224, 224)):
        """
        Khởi tạo ImageProcessor
        
        Args:
            target_size: Kích thước (width, height) cho ảnh đầu ra
        """
        self.target_size = target_size
        
    def preprocess_image(self, image):
        """
        Tiền xử lý ảnh để chuẩn bị cho model
        
        Args:
            image: Có thể là đường dẫn, numpy array, PIL Image, hoặc base64 string
            
        Returns:
            processed_img: Numpy array đã xử lý, shape (height, width, channels)
        """
        # Convert từ các định dạng khác nhau sang numpy array
        if isinstance(image, str):
            if image.startswith('data:image'):
                # Base64 image
                img = self.base64_to_numpy(image)
            else:
                # File path
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert từ BGR sang RGB
        elif isinstance(image, Image.Image):
            # PIL Image
            img = np.array(image)
        else:
            # Assume already numpy array
            img = image.copy()
            
        # Thay đổi kích thước
        img = cv2.resize(img, self.target_size)
        
        # Đảm bảo ảnh có 3 kênh màu RGB
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image
            img = img[:, :, :3]  # Bỏ kênh alpha
            
        return img
        
    def base64_to_numpy(self, base64_str):
        """Convert base64 string to numpy array"""
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
            
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img)
        
    def extract_faces(self, image, return_largest=True, min_confidence=0.9):
        """
        Trích xuất khuôn mặt từ ảnh
        
        Args:
            image: Ảnh đầu vào (numpy array)
            return_largest: Chỉ trả về khuôn mặt lớn nhất
            min_confidence: Ngưỡng tin cậy tối thiểu để phát hiện khuôn mặt
            
        Returns:
            faces: List các khuôn mặt phát hiện được
        """
        # Load pretrained face detector từ OpenCV DNN
        prototxt_path = "deploy.prototxt"  # Cần tải file này từ các nguồn mở
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"  # Cần tải file này từ các nguồn mở
        
        try:
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        except:
            print("Không thể tải face detector model. Hãy đảm bảo model files tồn tại.")
            # Fallback: trả về ảnh ban đầu đã resize
            img = self.preprocess_image(image)
            return [img] if return_largest else [img]
            
        # Chuẩn bị ảnh đầu vào
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
            
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
            
        # Phát hiện khuôn mặt
        net.setInput(blob)
        detections = net.forward()
        
        # Lọc và trích xuất khuôn mặt
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Đảm bảo kích thước hợp lệ
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Cắt khuôn mặt
                face = img[startY:endY, startX:endX]
                
                # Kiểm tra xem khuôn mặt có hợp lệ không
                if face.size > 0:
                    # Resize và thêm vào list
                    face_resized = cv2.resize(face, self.target_size)
                    faces.append({
                        'image': face_resized,
                        'confidence': confidence,
                        'box': (startX, startY, endX, endY)
                    })
                    
        # Sắp xếp theo kích thước lớn nhất (diện tích bounding box)
        if faces:
            faces = sorted(faces, key=lambda x: (x['box'][2] - x['box'][0]) * (x['box'][3] - x['box'][1]), reverse=True)
            
            if return_largest:
                return [faces[0]['image']]
            return [face['image'] for face in faces]
        else:
            # Không phát hiện khuôn mặt nào, dùng ảnh gốc
            img_resized = cv2.resize(img, self.target_size)
            return [img_resized]