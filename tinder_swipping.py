from selenium import webdriver  # Import thư viện webdriver để điều khiển trình duyệt
from selenium.webdriver.chrome.service import Service  # Import Service để cấu hình ChromeDriver
from selenium.webdriver.common.by import By  # Import By để tìm kiếm phần tử trên trang web
import credentials  # Import thông tin tài khoản từ file credentials.py
from time import sleep  # Import sleep để tạm dừng thực thi trong khoảng thời gian nhất định
from random import random, uniform  # Import random và uniform để tạo số ngẫu nhiên
import os  # Import thư viện os để thao tác với hệ thống tệp
import re  # Import thư viện re để sử dụng biểu thức chính quy
import requests  # Import requests để thực hiện HTTP requests tải ảnh
import uuid  # Import uuid để tạo mã định danh duy nhất
from datetime import datetime  # Import datetime để lấy thời gian hiện tại
import sys  # Import sys để thêm đường dẫn vào hệ thống
import numpy as np  # Import numpy để làm việc với arrays
import cv2  # Import OpenCV để xử lý ảnh
import tempfile  # Import tempfile để tạo file tạm thời
from PIL import Image
import io

# Thêm đường dẫn đến thư mục AI_Predict
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AI_Predict'))

# Import các module từ thư mục AI_Predict
try:
    from AI_Predict.PreferenceRecognition.preference_recognizer import PreferenceRecognizer
    from AI_Predict.PreferenceRecognition.image_processor import ImageProcessor
    from AI_Predict.PreferenceRecognition.model import PreferenceModel
    AI_AVAILABLE = True
except ImportError as e:
    print(f"Không thể import module AI Predict: {e}")
    AI_AVAILABLE = False

class EnsembleModel:
    """Lớp quản lý việc tải và dự đoán từ các mô hình ensemble"""
    
    def __init__(self):
        self.models = []
        self.model_paths = [
            os.path.join('AI_Predict','PreferenceRecognition' ,'saved_models', 'ensemble', 'preference_model_fold1.h5'),
            os.path.join('AI_Predict','PreferenceRecognition' , 'saved_models',  'ensemble', 'preference_model_fold2.h5'),
            os.path.join('AI_Predict','PreferenceRecognition' , 'saved_models',  'ensemble', 'preference_model_fold3.h5'),
            os.path.join('AI_Predict','PreferenceRecognition' , 'saved_models',  'ensemble', 'preference_model_fold4.h5'),
            os.path.join('AI_Predict','PreferenceRecognition' , 'saved_models', 'ensemble', 'preference_model_fold5.h5')
        ]
        
    def _load_models(self):
        """Tải các mô hình từ đường dẫn"""
        try:
            # Import tensorflow chỉ khi cần thiết để tránh lỗi nếu không có
            import tensorflow as tf
            import keras
            from keras.models import load_model
            
            # Kiểm tra xem các file mô hình có tồn tại không
            model_exists = False
            for model_path in self.model_paths:
                if os.path.exists(model_path):
                    model_exists = True
                    try:
                        model = load_model(model_path)
                        self.models.append(model)
                        print(f"Đã tải mô hình từ {model_path}")
                    except Exception as e:
                        print(f"Lỗi khi tải mô hình {model_path}: {e}")
            
            return len(self.models) > 0
        except ImportError:
            print("TensorFlow không được cài đặt. Sẽ sử dụng chế độ random.")
            return False
        
    def predict(self, image):
        """
        Dự đoán từ ảnh sử dụng mô hình ensemble
        
        Args:
            image: Numpy array ảnh đã tiền xử lý
            
        Returns:
            confidence: Độ tin cậy của dự đoán (0-1)
            predictions: Danh sách dự đoán của từng mô hình
        """
        if not self.models:
            return 0.5, []
            
        try:
            # Import numpy và tensorflow khi cần
            import tensorflow as tf
            import numpy as np
            
            # Chuẩn bị ảnh cho dự đoán
            img_array = np.expand_dims(image, axis=0).astype('float32') / 255.0
            
            # Dự đoán từ mỗi mô hình
            predictions = []
            for model in self.models:
                pred = model.predict(img_array, verbose=0)[0][0]  # Lấy giá trị dự đoán
                predictions.append(float(pred))
            
            # Tính trung bình độ tin cậy
            confidence = sum(predictions) / len(predictions)
            
            return confidence, predictions
            
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            return 0.5, []

class TinderBot():
    def __init__(self) -> None:
        driver_path = "./chromedriver-win64/chromedriver.exe"

        # Kiểm tra file chromedriver.exe có tồn tại không
        if not os.path.exists(driver_path):
            raise FileNotFoundError(f"Không tìm thấy chromedriver tại: {driver_path}")

        service = Service(
            executable_path = driver_path)
        options = webdriver.ChromeOptions()
        options.add_experimental_option("detach", True)
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.maximize_window()
        
        # Create folder for profile images
        self.image_folder = "tinder_profiles"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        print(f"Profile images will be saved in {os.path.abspath(self.image_folder)}")

        # Initialize AI model và image processor
        self.ensemble_model = EnsembleModel()
        self.image_processor = ImageProcessor()
        
        # Load ensemble model
        self.model_loaded = self.ensemble_model._load_models()
        if self.model_loaded:
            print(f"Đã tải thành công {len(self.ensemble_model.models)} mô hình ensemble cho dự đoán!")
        else:
            print("Không tìm thấy mô hình ensemble. Sẽ sử dụng chế độ random.")
            
        # Ngưỡng để dự đoán like (0-1)
        self.CONFIDENCE_THRESHOLD = 0.5
        
        # Create log file for AI decisions
        self.log_file = os.path.join("PreferenceRecognition", "logs", f"swipe_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write(f"Tinder AI Swipe Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("----------------------------------------\n")
            f.write("profile_id,num_images,prediction,confidence,decision,timestamp\n")


    def login(self) -> None:
        # Tinder
        self.driver.get("https://tinder.com")

        sleep(2)

        # Login 
        self.driver.find_element(
            By.XPATH, '//*[@id="s67002758"]/div/div[1]/div/main/div[1]/div/div/div/div/div/header/div/div[2]/div[2]/a').click()

        sleep(2)


        # open more options
       
        more_options_buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'More')]")
    
    
        if more_options_buttons:  # Nếu nút tồn tại
            more_options_buttons[0].click()  # Nhấn vào nút đầu tiên (nếu có nhiều nút)
        print("Opened more options")
        
        sleep(2)
        
#//*[@id="c-2079390956"]/div/div[1]/div/div[1]/div/div/div[2]/div[2]/span/button
#/html/body/div[2]/div/div[1]/div/div[1]/div/div/div[2]/div[2]/span/button

        # Connect with Facebook
        self.driver.find_element(
            By.XPATH, '//*[@id="s-1661378318"]/div/div[1]/div/div[1]/div/div/div[2]/div[2]/span/div[2]/button').click()
        

        sleep(2)

        # Switch popup window
        base_window = self.driver.window_handles[0]
        self.driver.switch_to.window(self.driver.window_handles[1])

        sleep(1)
        # Type email/password:
        self.driver.find_element(
            By.XPATH, '//*[@id="email"]').send_keys(credentials.EMAIL)
        self.driver.find_element(
            By.XPATH, '//*[@id="pass"]').send_keys(credentials.PASSWORD)

        sleep(1.5)
        self.driver.find_element(
            By.XPATH, '/html/body/div/div[2]/div[1]/form/div/div[3]/label[2]/input').click()
        sleep(5)
         # Handle Facebook CAPTCHA if it appears
        try:
            # Check if CAPTCHA is present
           
            if 0==0:
                print("CAPTCHA detected! Please solve it manually.")
                # Wait for manual CAPTCHA solving - adjust time as needed
                input("Press Enter after you've manually handled the login process...")
                
           
        except Exception as e:
            print(f"Facebook login process encountered an issue: {e}")
            print("Please check if the Facebook interface has changed or if CAPTCHA is present.")
            
            # Wait for manual intervention
            input("Press Enter after you've manually handled the login process...")

        sleep(4)
        self.driver.switch_to.window(base_window)
        sleep(2)

        # self.driver.find_element(
        #     By.XPATH, '//*[@id="o1172008979"]/div/div[1]/div/div/div[3]/button[1]/div[2]').click()
        
        # sleep(2)

        # self.driver.find_element(
        #     By.XPATH, '//*[@id="o1172008979"]/div/div[1]/div/div/div[3]/button[2]/div[2]').click()
        
        # sleep(2)
        # try:
        #     tinderPlusPopup =  self.driver.find_element(
        #         By.XPATH, '//*[@id="o1172008979"]/div/div[1]/div/div[3]/button[1]')
        #     if tinderPlusPopup :
        #         tinderPlusPopup.click()
        #         sleep(2)
        # except Exception as e:
        #     print("No Tinder Plus Popup")
            
        # self.driver.find_element(
        #     By.XPATH, '/html/body/div[1]/div/div[2]/div/div/div[1]/div[1]/button/div[2]').click()
        
        # sleep(3)

        
        # people_like = self.driver.find_elements(
        #         By.XPATH, '//*[@id="o1172008979"]/div/div/div/div[3]/button[2]')
            
        # if people_like:
        #     print("having people liking! Fck Them.")
        #     people_like.click()
        #     sleep(15)

        # self.driver.find_element(
        #     By.XPATH, '//*[@id="o697594959"]/main/div/div/div/div[3]/button[1]').click()
        # sleep(1)
        # self.driver.find_element(
        #     By.XPATH, '//*[@id="o697594959"]/main/div/div/div/div[3]/button[2]').click()

    def firstLike(self) -> bool:
        print("Attempting first like...")
        self.driver.find_element(
            By.XPATH, '//*[@id="main-content"]/div[1]/div/div/div/div[1]/div/div/div[4]/div/div[4]/button').click()
        # self.driver.find_element(
        #     By.XPATH, '//*[@id="main-content"]/div[1]/div/div/div/div[1]/div/div/div[4]/div/div[2]/button').click()
        print(" first like...")
        return True

    def superLike(self) -> bool:
        print("Attempting super like...")
        self.driver.find_element(
            By.XPATH, '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div/div[1]/div[1]/div/div[4]/div/div[3]/div/div/div/button').click()
        print(" super like ok...")
        return True

    def like(self) -> bool:
        
        xpath1 = '//*[@id="main-content"]/div[1]/div/div/div/div[1]/div/div/div[5]/div/div[4]/button'
        xpath2 = '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div/div[1]/div[1]/div/div[4]/div/div[4]/button'
        xpath3 = '//*[@id="c-351009880"]/div/div[1]/div/main/div[1]/div/div/div/div[1]/div[1]/div/div[3]/div/div[4]/button'
        print("Attempting like...")

        try:
            self.driver.find_element(By.XPATH, xpath1).click()
            return True
        except Exception as e:
            try:
                self.driver.find_element(By.XPATH, xpath2).click()
                print(" like ok...")
                return True
            except Exception:
                try:
                    self.driver.find_element(By.XPATH, xpath3).click()
                    print(" like ok...")
                    return True
                except Exception:
                    print("Error like!!!")
                    return False
    
    

    def dislike(self) -> bool:
        xpath1 = '//*[@id="main-content"]/div[1]/div/div/div/div[1]/div/div/div[4]/div/div[2]/button'
        xpath2 = '//*[@id="main-content"]/div[1]/div/div/div/div[1]/div/div/div[5]/div/div[2]/button'
        print("Attempting dislike...")
        xpath3 = '//*[@id="c-351009880"]/div/div[1]/div/main/div[1]/div/div/div/div[1]/div[1]/div/div[3]/div/div[2]/button'
        try:
            self.driver.find_element(By.XPATH, xpath1).click()
            print(" dislike ok...")
            return True
        except Exception as e:
            try:
                self.driver.find_element(By.XPATH, xpath2).click()
                print(" dislike ok...")
                return True
            except Exception:
                try:
                    self.driver.find_element(By.XPATH, xpath3).click()
                    print(" dislike ok...")
                    return True
                except Exception:
                    print("Error dislike!!!")
                    return False
        

    
    # Close Add to Home Popup window
    def add_home_popup(self) -> None:
        self.driver.find_element(
            By.XPATH, '/html/body/div[2]/div/div/div[2]/button[2]').click()
        # /html/body/div[2]/div/div/div[2]/button[2]

    # Close Match Popup window
    def match_popup(self) -> None:
        self.driver.find_element(
            By.XPATH, '//*[@id="o-1064206040"]/div/div/div[1]/div/div[3]/button').click()

    # Close Super Like Popup window
    def super_like_popup(self) -> None:
        self.driver.find_element(
            By.XPATH, '//*[@id="o697594959"]/main/div/button[2]').click()
        
    # Be seen later
    def be_seen_later(self) -> None:
        xpath1 = '/html/body/div[2]/div/div/button[2]'
        xpath2 = '//*[@id="c-2079390956"]/div/div/button[2]'
       
        try:
            self.driver.find_element(By.XPATH, xpath1).click()
        except Exception as e:
            try:
                self.driver.find_element(By.XPATH, xpath2).click()
            except Exception :
                print("Error be seen later!!!")
        
        #/html/body/div[2]/div/div/button[2]

    def handle_popups(self):
        try:
            self.add_home_popup()
        except Exception:
            try:
                self.match_popup()
            except Exception:
                try:
                    self.super_like_popup()
                except Exception:
                    try:
                        self.be_seen_later()
                    except Exception:
                        print("Errorrrr!!!")

    def extract_profile_image_url(self) -> str:
        """Extract the profile image URL from the current Tinder profile."""
        try:
            # Find the profile image div using the class attributes you provided
            profile_div = self.driver.find_element(By.CSS_SELECTOR, 'div.Bdrs\(8px\).Bgz\(cv\).Bgp\(c\).StretchedBox[style*="background-image"]')
            
            # Get the style attribute which contains the URL
            style_attr = profile_div.get_attribute('style')
            
            # Extract the URL using regex
            url_match = re.search(r'url\("([^"]+)"\)', style_attr)
            if url_match:
                image_url = url_match.group(1)
                print(f"Found profile image URL: {image_url[:60]}...")
                return image_url
            else:
                print("Could not extract image URL from style attribute")
                return None
        except Exception as e:
            print(f"Error extracting profile image: {e}")
            return None
    
    def save_profile_image(self) -> bool:
        """Save the current profile image to a file."""
        try:
            # Extract URL
            image_url = self.extract_profile_image_url()
            if not image_url:
                return False
            
            # Generate filename with timestamp and random ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{random_id}.jpg"
            filepath = os.path.join(self.image_folder, filename)
            
            # Download the image
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"Profile image saved to {filepath}")
                return True
            else:
                print(f"Failed to download image, status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error saving profile image: {e}")
            return False

    def extract_all_profile_images(self) -> list:
        """Trích xuất và lưu tất cả ảnh hồ sơ từ hồ sơ hiện tại đang hiển thị."""
        try:
            # QUAN TRỌNG: Trước tiên tìm thẻ hồ sơ đang hiển thị ở phía trước nhất
            # Tìm phần tử chứa nút like/dislike để đảm bảo đó là thẻ đang hiển thị phía trước
            main_card = None
            try:
                # Tìm phần tử nút Like (thường chỉ hiển thị cho thẻ phía trước nhất)
                like_buttons = self.driver.find_elements(By.XPATH, '//*[@id="main-content"]/div[1]/div/div/div/div[1]/div/div/div[5]/div/div[4]/button')
                if like_buttons:
                    # Đi lên vài cấp DOM từ nút Like để tìm thẻ hồ sơ
                    main_card = like_buttons[0]
                    for _ in range(5):  # Đi lên tối đa 5 cấp để tìm thẻ chứa hồ sơ
                        try:
                            parent = main_card.find_element(By.XPATH, '..')
                            main_card = parent
                            # Kiểm tra xem đây có phải là thẻ chứa hồ sơ không
                            if 'Pos(a)' in (main_card.get_attribute('class') or ''):
                                print("Đã tìm thấy thẻ hồ sơ chính dựa trên nút Like")
                                break
                        except Exception:
                            break
            except Exception as e:
                print(f"Không tìm thấy nút Like để xác định thẻ hồ sơ: {e}")
            
            # Nếu không tìm được từ nút Like, thử tìm từ container chính
            if not main_card:
                try:
                    # Tìm thẻ hồ sơ chính dựa trên cấu trúc DOM phổ biến
                    main_card = self.driver.find_element(By.XPATH, '//*[@id="main-content"]/div[1]/div/div/div/div[1]')
                    print("Đã tìm thấy thẻ hồ sơ chính dựa trên cấu trúc DOM")
                except Exception as e:
                    print(f"Không tìm thấy thẻ hồ sơ chính: {e}")
                    # Chụp screenshot để debug
                    self.driver.save_screenshot("card_not_found.png")
                    return []
            
            # Bước 1: Tìm tên người dùng từ thẻ hồ sơ chính để đảm bảo đúng người dùng đang hiển thị
            try:
                # Tìm theo cấu trúc DOM chính xác mà bạn đã cung cấp
                # Class Typs(display-1-strong) với thuộc tính itemprop="name"
                name_elements = main_card.find_elements(By.CSS_SELECTOR, 'span.Typs\\(display-1-strong\\)[itemprop="name"]')
                
                if not name_elements:
                    # Thử tìm bằng nhiều selector khác nhau
                    selectors = [
                        'span[itemprop="name"]',
                        'div.Ov\\(h\\) span[itemprop="name"]',
                        'div.Typs\\(display-1-strong\\) span',
                        'div[role="heading"]',
                        'h1'
                    ]
                    
                    for selector in selectors:
                        name_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if name_elements:
                            break
                
                if name_elements:
                    user_name = name_elements[1].text
                    print(f"Đã tìm thấy tên người dùng hiện tại: {user_name}")
                else:
                    print("Không tìm thấy tên người dùng trong thẻ hồ sơ")
                    # Chụp screenshot để debug
                    self.driver.save_screenshot("name_not_found.png")
                    user_name = None
            except Exception as e:
                print(f"Lỗi khi tìm tên người dùng: {e}")
                user_name = None
            
            # Bước 2: Từ thẻ hồ sơ chính, tìm carousel hiển thị ảnh
            current_carousel = None
            # try:
            #     # Tìm carousel trực tiếp trong thẻ hồ sơ chính đã xác định
            #     carousels = main_card.find_elements(By.CSS_SELECTOR, 'section[aria-roledescription="carousel"]')
            #     if carousels:
            #         # Lấy carousel đầu tiên trong thẻ hồ sơ chính
            #         current_carousel = carousels[0]
            #         carousel_label = current_carousel.get_attribute('aria-label')
            #         print(f"Đã tìm carousel trong thẻ hồ sơ: {carousel_label}")
            # except Exception as e:
            #     print(f"Lỗi khi tìm carousel trong thẻ hồ sơ: {e}")
            
            # # Nếu không tìm thấy carousel trong thẻ hồ sơ, thử tìm bằng tên người dùng
            # if not current_carousel and user_name:
            try:
                    # Tìm carousel dựa trên tên người dùng (ví dụ: aria-label="La La's photos")
                    current_carousel = self.driver.find_element(By.CSS_SELECTOR, f'section[aria-label="{user_name}\'s photos"]')
                    print(f"Đã tìm carousel của {user_name} bằng tên")
            except Exception as e:
                    print(f"Không tìm carousel bằng tên người dùng: {e}")
            
            # Nếu vẫn không tìm thấy carousel, kết thúc
            if not current_carousel:
                print("Không tìm thấy carousel cho hồ sơ hiện tại")
                # Chụp screenshot để debug
                self.driver.save_screenshot("carousel_not_found.png")
                return []
            
            # Xác minh rằng carousel thuộc về hồ sơ hiện tại (thường là thẻ hiển thị phía trên cùng)
            # Kiểm tra vị trí Z-index hoặc vị trí top của carousel
            try:
                parent_style = current_carousel.find_element(By.XPATH, "..").get_attribute("style")
                if "z-index: -1" in (parent_style or ""):
                    print("CẢNH BÁO: Carousel có thể thuộc về thẻ ở phía sau (z-index thấp)")
                    # Chụp screenshot để debug
                    self.driver.save_screenshot("wrong_carousel.png")
            except Exception:
                pass
            
            # Bước 3: Từ carousel, xác định tổng số ảnh
            tab_buttons = current_carousel.find_elements(By.CSS_SELECTOR, 'button[aria-label^="Photo"]')
            total_images = len(tab_buttons)
            
            if total_images == 0:
                print("Không tìm thấy ảnh nào trong carousel")
                return []
                
            print(f"Phát hiện {total_images} ảnh trong hồ sơ này")
            
            # Bước 4: Tìm các slide ảnh trong carousel
            slides = current_carousel.find_elements(By.CSS_SELECTOR, 'div.keen-slider__slide')
            if not slides:
                print("Không tìm thấy slide nào trong carousel")
                return []
            
            # Danh sách để lưu URL ảnh
            image_urls = []
            
            # Bước 5: Lấy URL ảnh đầu tiên (đã hiển thị sẵn)
            # Tìm slide đang hiển thị hiện tại (aria-hidden="false")
            current_slide = None
            for slide in slides:
                if slide.get_attribute('aria-hidden') == 'false':
                    current_slide = slide
                    break
            
            if not current_slide and slides:
                # Nếu không tìm thấy slide hiển thị, dùng slide đầu tiên
                current_slide = slides[0]
            
            if current_slide:
                try:
                    # Tìm phần tử div chứa background-image
                    image_div = current_slide.find_element(By.CSS_SELECTOR, 'div[style*="background-image"]')
                    style_attr = image_div.get_attribute('style')
                    url_match = re.search(r'url\("([^"]+)"\)', style_attr)
                    
                    if url_match:
                        image_urls.append(url_match.group(1))
                        print(f"Đã trích xuất URL ảnh đầu tiên: {url_match.group(1)[:50]}...")
                    else:
                        print("Không thể trích xuất URL từ style của ảnh đầu tiên")
                except Exception as e:
                    print(f"Lỗi khi trích xuất ảnh đầu tiên: {e}")
            
            # Nếu chỉ có 1 ảnh, trả về kết quả
            if total_images <= 1:
                return image_urls
            
            # Bước 6: Click nút Next để xem và lưu các ảnh tiếp theo
            try:
                
                
                # Click qua từng ảnh và lưu URL
                for i in range(1, total_images):
                    try:
                        try:
                            # Tìm nút Next trong carousel
                            next_button = self.driver.find_element(By.XPATH, '//*[@id="main-content"]/div[1]/div/div/div/div[1]/div/div/div[2]/div[1]/section/button[2]')
                            # Click nút Next
                            next_button.click()
                        except Exception as e:
                            print(f"Error clicking Next button for photo {i+1}: {e}")
                            try:
                                # Attempt to find an alternative Next button and click again
                                next_button = current_carousel.find_element(By.CSS_SELECTOR, '//button[contains(@aria-label, "Next Photo")]')
                                next_button.click()
                            except Exception as e2:
                                print(f"Failed to find or click alternative Next button for photo {i+1}: {e2}")
                                break
                        print(f"Đã click nút Next để chuyển đến ảnh {i+1}/{total_images}")
                        
                        # Đợi ảnh tải
                        sleep(0.5)
                        
                        # Tìm slide hiện tại (aria-hidden="false" sau khi click)
                        current_slide = None
                        for slide in slides:
                            if slide.get_attribute('aria-hidden') == 'false':
                                current_slide = slide
                                break
                        
                        if current_slide:
                            try:
                                # Lấy URL ảnh hiện tại
                                image_div = current_slide.find_element(By.CSS_SELECTOR, 'div[style*="background-image"]')
                                style_attr = image_div.get_attribute('style')
                                url_match = re.search(r'url\("([^"]+)"\)', style_attr)
                                
                                if url_match:
                                    image_urls.append(url_match.group(1))
                                    print(f"Đã trích xuất URL ảnh {i+1}/{total_images}")
                                else:
                                    print(f"Không thể trích xuất URL từ style của ảnh {i+1}")
                            except Exception as e:
                                print(f"Lỗi khi trích xuất ảnh {i+1}: {e}")
                        else:
                            print(f"Không tìm thấy slide đang hiển thị sau khi click Next cho ảnh {i+1}")
                    except Exception as e:
                        print(f"Lỗi khi điều hướng đến ảnh {i+1}: {e}")
                
                # Bước 7: Quay lại ảnh đầu tiên
                try:
                    # Tìm nút Previous
                    prev_button = current_carousel.find_element(By.CSS_SELECTOR, 'button[aria-label="Previous Photo"]')
                    
                    # Click nút Previous để quay lại đầu
                    for _ in range(total_images - 1):
                        prev_button.click()
                        sleep(0.2)
                    
                    print("Đã quay về ảnh đầu tiên")
                except Exception as e:
                    print(f"Lỗi khi quay về ảnh đầu tiên: {e}")
                
            except Exception as e:
                print(f"Không tìm thấy nút Next để chuyển đến các ảnh tiếp theo: {e}")
            
            return image_urls
            
        except Exception as e:
            print(f"Lỗi tổng thể khi trích xuất ảnh hồ sơ: {e}")
            # Chụp screenshot để debug
            self.driver.save_screenshot("error_extract_images.png")
            print("Đã lưu screenshot để debug")
            return []

    
    def create_prediction_visualization(self, image_url, confidence, decision, i, total, profile_id):
        """Tạo hình ảnh trực quan hóa kết quả dự đoán AI và lưu lại"""
        try:
            # Tải ảnh từ URL
            response = requests.get(image_url, stream=True)
            if response.status_code != 200:
                print(f"Không thể tải ảnh để trực quan hóa")
                return
            
            # Đọc ảnh bằng OpenCV
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
            # Kích thước ảnh
            height, width = img.shape[:2]
        
            # Thêm overlay nền cho phần thông tin
            overlay = img.copy()
            cv2.rectangle(overlay, (0, height-150), (width, height), (0, 0, 0), -1)
            alpha = 0.7  # Độ trong suốt
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
            # Thông tin dự đoán để hiển thị
            info_text = [
                f"Profile ID: {profile_id}",
                f"Image: {i+1}/{total}",
                f"Confidence: {confidence:.3f}",
                f"Threshold: {self.CONFIDENCE_THRESHOLD:.2f}",
                f"Decision: {'LIKE' if decision else 'DISLIKE'}",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]
        
            # Màu chữ dựa theo quyết định
            text_color = (0, 255, 0) if decision else (0, 0, 255)  # Xanh lá cho LIKE, đỏ cho DISLIKE
        
            # Thêm text lên ảnh
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            for idx, text in enumerate(info_text):
                y_pos = height - 130 + (idx * 25)
                cv2.putText(img, text, (20, y_pos), font, font_scale, text_color, 2)
        
            # Tạo thư mục lưu ảnh AI log nếu chưa có
            ai_log_folder = os.path.join(self.image_folder, "ai_logs")
            if not os.path.exists(ai_log_folder):
                os.makedirs(ai_log_folder)
        
            # Tạo tên file ảnh log với thông tin quyết định
            decision_text = "LIKE" if decision else "DISLIKE"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{profile_id}_img{i+1}of{total}_{confidence:.2f}_{decision_text}.jpg"
            filepath = os.path.join(ai_log_folder, filename)
        
            # Lưu ảnh
            cv2.imwrite(filepath, img)
            print(f"Đã lưu ảnh trực quan hóa AI tại: {filepath}")
        
            return filepath
        
        except Exception as e:
            print(f"Lỗi khi tạo trực quan hóa dự đoán: {e}")
            return None
        
    def predict_preference(self, image_urls):
        if not self.model_loaded or not image_urls:
            return random() < 0.8, 0.5, []
            
        predictions = []
        confidences = []
        profile_id = str(uuid.uuid4())[:8]  # ID duy nhất cho hồ sơ này
        total_images = len(image_urls)
        visualization_paths = []
        
        for i, url in enumerate(image_urls):
            try:
                response = requests.get(url, stream=True)
                if response.status_code != 200:
                    print(f"Không thể tải ảnh {i+1}/{len(image_urls)}")
                    continue
                    
                img = Image.open(io.BytesIO(response.content))
                img_array = np.array(img)
                
                processed_img = self.image_processor.preprocess_image(img_array)
                
                confidence, individual_preds = self.ensemble_model.predict(processed_img)
                
                # Quyết định cho ảnh hiện tại
                image_decision = confidence >= self.CONFIDENCE_THRESHOLD
                
                # Tạo trực quan hóa kết quả AI cho ảnh này
                vis_path = self.create_prediction_visualization(
                    url, confidence, image_decision, i, total_images, profile_id
                )
                if vis_path:
                    visualization_paths.append(vis_path)
                
                predictions.append(individual_preds)
                confidences.append(confidence)
                
                print(f"Ảnh {i+1}/{len(image_urls)}: Dự đoán với độ tin cậy {confidence:.2f} -> {'LIKE' if image_decision else 'DISLIKE'}")
                
                # Log chi tiết từng mô hình
                if individual_preds:
                    for j, pred in enumerate(individual_preds):
                        print(f"  - Mô hình {j+1}: {pred:.3f}")
                
            except Exception as e:
                print(f"Lỗi khi dự đoán ảnh {i+1}: {e}")
        
        if not confidences:
            return random() < 0.8, 0.5, []
        
        avg_confidence = sum(confidences) / len(confidences)
        
        decision = avg_confidence >= self.CONFIDENCE_THRESHOLD
        
        # Log AI summary in the console
        print("\n===== TÓM TẮT KẾT QUẢ AI =====")
        print(f"• Tổng hồ sơ ({profile_id}): {len(image_urls)} ảnh")
        print(f"• Độ tin cậy trung bình: {avg_confidence:.3f}")
        print(f"• Kết quả từng ảnh: {[f'{c:.2f}' for c in confidences]}")
        print(f"• Quyết định cuối cùng: {'LIKE' if decision else 'DISLIKE'}")
        print("==============================\n")
        
        return decision, avg_confidence, predictions
    

    def save_all_profile_images(self) -> bool:
        """Save all images from the current profile."""
        image_urls = self.extract_all_profile_images()
        
        if not image_urls:
            print("No profile images found to save")
            return False
        
        # Generate a unique ID for this profile
        profile_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all images
        for i, url in enumerate(image_urls):
            try:
                # Create filename with profile_id to group images from same profile
                filename = f"{timestamp}__{profile_id}_photo{i+1}of{len(image_urls)}.jpg"
                filepath = os.path.join(self.image_folder, filename)
                
                # Download image
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    print(f"Saved profile image {i+1}/{len(image_urls)} to {filepath}")
                else:
                    print(f"Failed to download image {i+1}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error saving image {i+1}: {e}")
        
        print(f"Saved {len(image_urls)} images for profile {profile_id}")
        return True

    # Auto Switch feature
    def swipe(self) -> None:
        # # refresh page
        # self.driver.refresh()
        # sleep(4)
        # Lưu tất cả ảnh hồ sơ trước khi thực hiện thao tác like/dislike
        try:
            self.save_all_profile_images()
        except Exception as e:
            print(f"Không thể lưu ảnh hồ sơ: {e}")    
        # Must call on first swipe
        self.firstLike()
        sleep(1)

        likeCount, dislikeCount = 0, 0

        while likeCount + dislikeCount < 15:
            if (likeCount + dislikeCount) % 10 == 0:
                print(f"Total Like: {likeCount} , Total Dislike: {dislikeCount}")

            sleep(uniform(0.5, 1.5))
            
            try:
                # Bước 1: Lấy tất cả ảnh từ hồ sơ hiện tại
                print("Trích xuất ảnh từ hồ sơ hiện tại...")
                image_urls = self.extract_all_profile_images()
                
                # Log thông tin hồ sơ
                total_images = len(image_urls)
                profile_id = str(uuid.uuid4())[:8]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Bước 2: Lưu tất cả ảnh vào thư mục
                for i, url in enumerate(image_urls):
                    try:
                        # Tạo tên file để lưu ảnh
                        filename = f"{timestamp}_{profile_id}_photo{i+1}of{total_images}.jpg"
                        filepath = os.path.join(self.image_folder, filename)
                        
                        # Tải và lưu ảnh
                        response = requests.get(url, stream=True)
                        if response.status_code == 200:
                            with open(filepath, 'wb') as f:
                                for chunk in response.iter_content(1024):
                                    f.write(chunk)
                            print(f"Đã lưu ảnh {i+1}/{total_images} vào {filepath}")
                        else:
                            print(f"Không thể tải ảnh {i+1}, status code: {response.status_code}")
                    except Exception as e:
                        print(f"Lỗi khi lưu ảnh {i+1}: {e}")
                
                # Bước 3: Dự đoán sở thích dựa trên ảnh
                decision, confidence, predictions = self.predict_preference(image_urls)
                
                # Ghi log quyết định
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"{profile_id},{total_images},{decision},{confidence:.3f},{'like' if decision else 'dislike'},{timestamp}\n")
                    
                # Hiển thị thông tin dự đoán
                print(f"\n===== THÔNG TIN DỰ ĐOÁN =====")
                print(f"• Hồ sơ ID: {profile_id} | Số ảnh: {total_images}")
                print(f"• Độ tin cậy: {confidence:.2f} | Ngưỡng: {self.CONFIDENCE_THRESHOLD:.2f}")
                print(f"• Quyết định: {'LIKE' if decision else 'DISLIKE'} dựa trên {total_images} ảnh")
                print(f"==========================\n")
                
                # Bước 4: Thực hiện quyết định swipe dựa trên dự đoán
                if decision:
                    # Quyết định LIKE
                    if confidence > 0.85 and random() < 0.2:  # Đôi khi Super Like nếu confidence rất cao
                        if self.superLike():
                            print("Đã Super Like dựa trên dự đoán cao")
                            likeCount += 1
                    else:
                        if self.like():
                            print("Đã Like dựa trên dự đoán AI")
                            likeCount += 1
                        else:
                            print("Không thể Like, xử lý các popup...")
                            self.handle_popups()
                else:
                    # Quyết định DISLIKE
                    if self.dislike():
                        print("Đã Dislike dựa trên dự đoán AI")
                        dislikeCount += 1
                    else:
                        print("Không thể Dislike, xử lý các popup...")
                        self.handle_popups()

                print(f"Tổng swipe: {likeCount + dislikeCount} (Like: {likeCount}, Dislike: {dislikeCount})")
                sleep(uniform(0.5, 1.5))  # Chờ một chút sau khi thực hiện quyết định

            except Exception as e:
                print(f"Lỗi không mong muốn xảy ra: {e}")
                self.handle_popups()  # Thử xử lý các popup nếu có lỗi
                sleep(1)  # Đợi thêm nếu xảy ra lỗi

        print(f'Báo cáo cuối cùng ({likeCount+dislikeCount} swipes): \n • Like: {likeCount} \n • Dislike: {dislikeCount}')


bot = TinderBot()
bot.login()
# Wait facebook login redirect to main page
sleep(7)
bot.swipe()
