import requests
import json
import time
from datetime import datetime
import os

class TinderClient:
    """
    Client để tương tác với Tinder API không chính thức
    Lưu ý: Sử dụng không chính thức API có thể dẫn đến việc tài khoản bị khóa
    """
    
    BASE_URL = "https://api.gotinder.com"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    
    def __init__(self, auth_token=None, token_file=None):
        """
        Khởi tạo Tinder client
        
        Args:
            auth_token: X-Auth-Token của Tinder (có thể lấy từ web app)
            token_file: File lưu token (nếu không cung cấp auth_token trực tiếp)
        """
        self.headers = {
            "User-Agent": self.USER_AGENT,
            "Content-Type": "application/json",
            "X-Auth-Token": auth_token
        }
        
        self.token_file = token_file or os.path.join(
            os.path.dirname(__file__), 
            'credentials', 
            'tinder_token.json'
        )
        
        # Tạo thư mục lưu token nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
        
        # Nếu không cung cấp auth_token, thử đọc từ file
        if not auth_token:
            self._load_token()
            
    def _load_token(self):
        """Tải token từ file lưu trữ"""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as f:
                    data = json.load(f)
                    if 'token' in data:
                        self.headers["X-Auth-Token"] = data['token']
                        return True
        except Exception as e:
            print(f"Lỗi khi đọc token: {e}")
        return False
        
    def _save_token(self, token):
        """Lưu token vào file"""
        try:
            with open(self.token_file, 'w') as f:
                json.dump({'token': token, 'updated_at': datetime.now().isoformat()}, f)
            return True
        except Exception as e:
            print(f"Lỗi khi lưu token: {e}")
            return False
            
    def set_auth_token(self, token):
        """
        Thiết lập auth token
        
        Args:
            token: X-Auth-Token
            
        Returns:
            success: True nếu thành công
        """
        self.headers["X-Auth-Token"] = token
        return self._save_token(token)
        
    def get_recs(self):
        """
        Lấy danh sách recommendations (người dùng để swipe)
        
        Returns:
            recs: List các profile được đề xuất
        """
        url = f"{self.BASE_URL}/v2/recs/core"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", {}).get("results", [])
            else:
                print(f"Lỗi khi lấy recommendations: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception khi lấy recommendations: {e}")
            return []
            
    def like(self, user_id):
        """
        Swipe right (like) một người dùng
        
        Args:
            user_id: ID của người dùng
            
        Returns:
            match: True nếu match, False nếu không
        """
        url = f"{self.BASE_URL}/like/{user_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                return data.get("match", False)
            else:
                print(f"Lỗi khi like user {user_id}: {response.status_code}")
                return False
        except Exception as e:
            print(f"Exception khi like user {user_id}: {e}")
            return False
            
    def dislike(self, user_id):
        """
        Swipe left (pass) một người dùng
        
        Args:
            user_id: ID của người dùng
            
        Returns:
            success: True nếu thành công
        """
        url = f"{self.BASE_URL}/pass/{user_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            return response.status_code == 200
        except Exception as e:
            print(f"Exception khi dislike user {user_id}: {e}")
            return False
            
    def get_profile(self, user_id):
        """
        Lấy thông tin chi tiết của một profile
        
        Args:
            user_id: ID của người dùng
            
        Returns:
            profile: Thông tin profile
        """
        url = f"{self.BASE_URL}/user/{user_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json().get("results")
            else:
                print(f"Lỗi khi lấy profile {user_id}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception khi lấy profile {user_id}: {e}")
            return None
            
    def get_self_profile(self):
        """
        Lấy thông tin profile của bản thân
        
        Returns:
            profile: Thông tin profile của bản thân
        """
        url = f"{self.BASE_URL}/profile"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Lỗi khi lấy profile: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception khi lấy profile: {e}")
            return None
            
    def extract_profile_images(self, profile):
        """
        Trích xuất URLs ảnh từ profile data
        
        Args:
            profile: Dữ liệu profile từ API
            
        Returns:
            images: List các URL của ảnh
        """
        images = []
        
        if isinstance(profile, dict) and "photos" in profile:
            for photo in profile["photos"]:
                if "url" in photo:
                    images.append(photo["url"])
                elif "processedFiles" in photo:
                    # Lấy ảnh có độ phân giải cao nhất
                    for processed in sorted(
                        photo["processedFiles"], 
                        key=lambda x: x.get("width", 0) * x.get("height", 0),
                        reverse=True
                    ):
                        if "url" in processed:
                            images.append(processed["url"])
                            break
        
        return images