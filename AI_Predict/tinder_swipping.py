from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import credentials
from time import sleep
from random import random, uniform
import os
import re
import requests
import uuid
from datetime import datetime
import numpy as np
from PIL import Image
import io

# Import AI preference recognition modules
from PreferenceRecognition.model import EnsembleModel
from PreferenceRecognition.image_processor import ImageProcessor

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
        
        # Create AI preference folder
        self.ai_preference_folder = os.path.join(self.image_folder, "ai_decisions")
        if not os.path.exists(self.ai_preference_folder):
            os.makedirs(self.ai_preference_folder)
            os.makedirs(os.path.join(self.ai_preference_folder, "like"))
            os.makedirs(os.path.join(self.ai_preference_folder, "dislike"))
        
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
        self.CONFIDENCE_THRESHOLD = 0.6
        
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
            By.XPATH, '//*[@id="o-1394577241"]/div/div[1]/div/main/div[1]/div/div/div/div/div/header/div/div[2]/div[2]/a/div[2]').click()

        sleep(2)


        # open more options
       
        more_options_buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'More')]")
    
    
        if more_options_buttons:  # Nếu nút tồn tại
            more_options_buttons[0].click()  # Nhấn vào nút đầu tiên (nếu có nhiều nút)
        print("Opened more options")
        
        sleep(2)
        
        # Connect with Facebook
        self.driver.find_element(
            By.XPATH, '//*[@id="o1172008979"]/div/div[1]/div/div[1]/div/div/div[2]/div[2]/span/div[2]/button/div[2]').click()
        

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

        self.driver.find_element(
            By.XPATH, '//*[@id="o1172008979"]/div/div[1]/div/div/div[3]/button[1]/div[2]').click()
        
        sleep(2)

        self.driver.find_element(
            By.XPATH, '//*[@id="o1172008979"]/div/div[1]/div/div/div[3]/button[2]/div[2]').click()
        
        sleep(2)
        try:
            tinderPlusPopup =  self.driver.find_element(
                By.XPATH, '//*[@id="o1172008979"]/div/div[1]/div/div[3]/button[1]')
            if tinderPlusPopup :
                tinderPlusPopup.click()
                sleep(2)
        except Exception as e:
            print("No Tinder Plus Popup")
            
        self.driver.find_element(
            By.XPATH, '/html/body/div[1]/div/div[2]/div/div/div[1]/div[1]/button/div[2]').click()
        
        sleep(3)

        
        people_like = self.driver.find_elements(
                By.XPATH, '//*[@id="o1172008979"]/div/div/div/div[3]/button[2]')
            
        if people_like:
            print("having people liking! Fck Them.")
            people_like.click()
            sleep(15)

    def firstLike(self) -> bool:
        print("Attempting first like...")
        self.driver.find_element(
            By.XPATH, '//*[@id="main-content"]/div[1]/div/div/div/div[1]/div/div/div[4]/div/div[4]/button').click()
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
            profile_div = self.driver.find_element(By.CSS_SELECTOR, 'div.Bdrs\(8px\).Bgz\(cv\).Bgp\(c\).StretchedBox[style*="background-image"]')
            style_attr = profile_div.get_attribute('style')
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
    
    def extract_all_profile_images(self) -> list:
        try:
            main_card = None
            try:
                like_buttons = self.driver.find_elements(By.XPATH, '//*[@id="main-content"]/div[1]/div/div/div/div[1]/div/div/div[5]/div/div[4]/button')
                if like_buttons:
                    main_card = like_buttons[0]
                    for _ in range(5):
                        try:
                            parent = main_card.find_element(By.XPATH, '..')
                            main_card = parent
                            if 'Pos(a)' in (main_card.get_attribute('class') or ''):
                                print("Đã tìm thấy thẻ hồ sơ chính dựa trên nút Like")
                                break
                        except Exception:
                            break
            except Exception as e:
                print(f"Không tìm thấy nút Like để xác định thẻ hồ sơ: {e}")
            
            if not main_card:
                try:
                    main_card = self.driver.find_element(By.XPATH, '//*[@id="main-content"]/div[1]/div/div/div/div[1]')
                    print("Đã tìm thấy thẻ hồ sơ chính dựa trên cấu trúc DOM")
                except Exception as e:
                    print(f"Không tìm thấy thẻ hồ sơ chính: {e}")
                    self.driver.save_screenshot("card_not_found.png")
                    return []
            
            try:
                name_elements = main_card.find_elements(By.CSS_SELECTOR, 'span.Typs\\(display-1-strong\\)[itemprop="name"]')
                
                if not name_elements:
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
                    self.driver.save_screenshot("name_not_found.png")
                    user_name = None
            except Exception as e:
                print(f"Lỗi khi tìm tên người dùng: {e}")
                user_name = None
            
            current_carousel = None
            try:
                current_carousel = self.driver.find_element(By.CSS_SELECTOR, f'section[aria-label="{user_name}\'s photos"]')
                print(f"Đã tìm carousel của {user_name} bằng tên")
            except Exception as e:
                print(f"Không tìm carousel bằng tên người dùng: {e}")
            
            if not current_carousel:
                print("Không tìm thấy carousel cho hồ sơ hiện tại")
                self.driver.save_screenshot("carousel_not_found.png")
                return []
            
            try:
                parent_style = current_carousel.find_element(By.XPATH, "..").get_attribute("style")
                if "z-index: -1" in (parent_style or ""):
                    print("CẢNH BÁO: Carousel có thể thuộc về thẻ ở phía sau (z-index thấp)")
                    self.driver.save_screenshot("wrong_carousel.png")
            except Exception:
                pass
            
            tab_buttons = current_carousel.find_elements(By.CSS_SELECTOR, 'button[aria-label^="Photo"]')
            total_images = len(tab_buttons)
            
            if total_images == 0:
                print("Không tìm thấy ảnh nào trong carousel")
                return []
                
            print(f"Phát hiện {total_images} ảnh trong hồ sơ này")
            
            slides = current_carousel.find_elements(By.CSS_SELECTOR, 'div.keen-slider__slide')
            if not slides:
                print("Không tìm thấy slide nào trong carousel")
                return []
            
            image_urls = []
            
            current_slide = None
            for slide in slides:
                if slide.get_attribute('aria-hidden') == 'false':
                    current_slide = slide
                    break
            
            if not current_slide and slides:
                current_slide = slides[0]
            
            if current_slide:
                try:
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
            
            if total_images <= 1:
                return image_urls
            
            try:
                for i in range(1, total_images):
                    try:
                        try:
                            next_button = self.driver.find_element(By.XPATH, '//*[@id="main-content"]/div[1]/div/div/div/div[1]/div/div/div[2]/div[1]/section/button[2]')
                            next_button.click()
                        except Exception as e:
                            print(f"Error clicking Next button for photo {i+1}: {e}")
                            try:
                                next_button = current_carousel.find_element(By.CSS_SELECTOR, '//button[contains(@aria-label, "Next Photo")]')
                                next_button.click()
                            except Exception as e2:
                                print(f"Failed to find or click alternative Next button for photo {i+1}: {e2}")
                                break
                        print(f"Đã click nút Next để chuyển đến ảnh {i+1}/{total_images}")
                        
                        sleep(0.5)
                        
                        current_slide = None
                        for slide in slides:
                            if slide.get_attribute('aria-hidden') == 'false':
                                current_slide = slide
                                break
                        
                        if current_slide:
                            try:
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
                
                try:
                    prev_button = current_carousel.find_element(By.CSS_SELECTOR, 'button[aria-label="Previous Photo"]')
                    
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
            self.driver.save_screenshot("error_extract_images.png")
            print("Đã lưu screenshot để debug")
            return []

    def predict_preference(self, image_urls):
        if not self.model_loaded or not image_urls:
            return random() < 0.8, 0.5, []
            
        predictions = []
        confidences = []
        
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
                
                predictions.append(individual_preds)
                confidences.append(confidence)
                
                print(f"Ảnh {i+1}/{len(image_urls)}: Dự đoán với độ tin cậy {confidence:.2f}")
                
            except Exception as e:
                print(f"Lỗi khi dự đoán ảnh {i+1}: {e}")
        
        if not confidences:
            return random() < 0.8, 0.5, []
        
        avg_confidence = sum(confidences) / len(confidences)
        
        decision = avg_confidence >= self.CONFIDENCE_THRESHOLD
        
        return decision, avg_confidence, predictions

    def save_all_profile_images(self) -> dict:
        image_urls = self.extract_all_profile_images()
        
        if not image_urls:
            print("No profile images found to save")
            return {"decision": random() < 0.8, "confidence": 0, "image_count": 0, "profile_id": None}
        
        profile_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ai_decision, confidence, predictions = self.predict_preference(image_urls)
        
        with open(self.log_file, "a") as f:
            f.write(f"{profile_id},{len(image_urls)},{ai_decision},{confidence:.4f},{ai_decision},{timestamp}\n")
            
        print(f"AI {'LIKES' if ai_decision else 'DISLIKES'} this profile with confidence: {confidence:.4f}")
        
        ai_folder = os.path.join(self.ai_preference_folder, "like" if ai_decision else "dislike")
        
        for i, url in enumerate(image_urls):
            try:
                filename = f"{timestamp}_{profile_id}_photo{i+1}of{len(image_urls)}.jpg"
                filepath = os.path.join(self.image_folder, filename)
                
                ai_filepath = os.path.join(ai_folder, filename)
                
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                            
                    with open(ai_filepath, 'wb') as f:
                        response = requests.get(url, stream=True)
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                            
                    print(f"Saved profile image {i+1}/{len(image_urls)} to {filepath}")
                else:
                    print(f"Failed to download image {i+1}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error saving image {i+1}: {e}")
        
        print(f"Saved {len(image_urls)} images for profile {profile_id}")
        
        return {
            "decision": ai_decision, 
            "confidence": confidence, 
            "image_count": len(image_urls),
            "profile_id": profile_id
        }

    def swipe(self) -> None:
        self.firstLike()
        sleep(1)

        likeCount, dislikeCount = 0, 0

        while likeCount + dislikeCount < 1000:
            if (likeCount + dislikeCount) % 100 == 0:
                print(f"Total Like: {likeCount}, Total Dislike: {dislikeCount}")

            sleep(uniform(0.5, 2.9))
            
            try:
                try:
                    result = self.save_all_profile_images()
                    ai_likes = result["decision"]
                    ai_confidence = result["confidence"]
                    profile_id = result["profile_id"]
                    
                    print(f"Profile {profile_id}: AI {'LIKES' if ai_likes else 'DISLIKES'} with {ai_confidence:.2f} confidence")
                    
                    if ai_likes:
                        if self.like():
                            likeCount += 1
                            print(f"✅ LIKED profile based on AI prediction")
                        else:
                            self.handle_popups()
                    else:
                        if self.dislike():
                            dislikeCount += 1
                            print(f"❌ DISLIKED profile based on AI prediction")
                        else:
                            self.handle_popups()
                            
                except Exception as e:
                    print(f"Lỗi khi dự đoán và xử lý ảnh hồ sơ: {e}")
                    randomNum = random()
                    if randomNum < 0.65:
                        if self.like():
                            likeCount += 1
                            print("LIKED profile (random fallback)")
                        else:
                            self.handle_popups()
                    else:
                        if self.dislike():
                            dislikeCount += 1
                            print("DISLIKED profile (random fallback)")
                        else:
                            self.handle_popups()

                print(f"Total Swipe: {likeCount + dislikeCount}")

            except Exception as e:
                print(f"Unexpected error occurred: {e}")

        print(
            f'Final Report ({likeCount+dislikeCount} swipes): \n Likes: {likeCount} \n Dislikes: {dislikeCount}')


bot = TinderBot()
bot.login()
sleep(7)
bot.swipe()
