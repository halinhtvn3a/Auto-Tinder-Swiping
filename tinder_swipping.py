from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import credentials
from time import sleep
from random import random, uniform


class TinderBot():
    def __init__(self) -> None:
        service = Service(
            executable_path="./chromedriver-win64/chromedriver.exe")
        options = webdriver.ChromeOptions()
        options.add_experimental_option("detach", True)
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.maximize_window()

    def login(self) -> None:
        # Tinder
        self.driver.get("https://tinder.com")

        sleep(2)

        # Login 
        self.driver.find_element(
            By.XPATH, '//*[@id="c-351009880"]/div/div[1]/div/main/div[1]/div/div/div/div/div/header/div/div[2]/div[2]/a').click()

        sleep(2)


        # open more options
       
        more_options_buttons = self.driver.find_elements(
        By.XPATH, '//*[@id="c-2079390956"]/div/div[1]/div/div[1]/div/div/div[2]/div[2]/span/button'
        )   
    
        if more_options_buttons:  # Nếu nút tồn tại
            more_options_buttons[0].click()  # Nhấn vào nút đầu tiên (nếu có nhiều nút)
        print("Opened more options")
        
        sleep(2)
        
#//*[@id="c-2079390956"]/div/div[1]/div/div[1]/div/div/div[2]/div[2]/span/button
#/html/body/div[2]/div/div[1]/div/div[1]/div/div/div[2]/div[2]/span/button

        # Connect with Facebook
        self.driver.find_element(
            By.XPATH, '//*[@id="c-2079390956"]/div/div[1]/div/div[1]/div/div/div[2]/div[2]/span/div[2]/button').click()
        

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
        self.driver.find_element(
            By.XPATH, '/html/body/div[1]/div/div/div/div/div/div/div[1]/div[3]/div/div/div/div/div/div/div[2]/div/div/div[1]/div/div/div/div[1]/div/div/div/div').click()

        sleep(4)
        self.driver.switch_to.window(base_window)
        sleep(2)

        self.driver.find_element(
            By.XPATH, '//*[@id="c-2079390956"]/div/div[1]/div/div/div[3]/button[1]').click()
        
        sleep(2)

        self.driver.find_element(
            By.XPATH, '//*[@id="c-2079390956"]/div/div[1]/div/div/div[3]/button[2]').click()
        
        sleep(2)

        self.driver.find_element(
            By.XPATH, '//*[@id="c-351009880"]/div/div[2]/div/div/div[1]/div[1]/button').click()
        
        sleep(1)
        # self.driver.find_element(
        #     By.XPATH, '//*[@id="o697594959"]/main/div/div/div/div[3]/button[1]').click()
        # sleep(1)
        # self.driver.find_element(
        #     By.XPATH, '//*[@id="o697594959"]/main/div/div/div/div[3]/button[2]').click()

    def firstLike(self) -> bool:
        print("Attempting first like...")
        self.driver.find_element(
            By.XPATH, '//*[@id="c-351009880"]/div/div[1]/div/main/div[1]/div/div/div/div[1]/div[1]/div/div[3]/div/div[4]/button').click()
        print(" first like...")
        return True

    def superLike(self) -> bool:
        print("Attempting super like...")
        self.driver.find_element(
            By.XPATH, '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div/div[1]/div[1]/div/div[4]/div/div[3]/div/div/div/button').click()
        print(" super like ok...")
        return True

    def like(self) -> bool:
        xpath1 = '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div/div[1]/div[1]/div/div[5]/div/div[4]/button'
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
        xpath1 = '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div/div[1]/div[1]/div/div[5]/div/div[2]/button'
        xpath2 = '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div/div[1]/div[1]/div/div[4]/div/div[2]/button'
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
            By.XPATH, '/html/body/div[1]/div/div[1]/div/main/div[2]/div/div/div[1]/div/div[4]/button').click()

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

    # Auto Switch feature
    def swipe(self) -> None:
        # Must call on first swipe
        self.firstLike()
        sleep(1)

        likeCount, dislikeCount = 0, 0

        while likeCount + dislikeCount < 1000:
            if (likeCount + dislikeCount) % 100 == 0 :
                print(f"Total Like: {likeCount} , Total Dislike: {dislikeCount}")

            sleep(uniform(0.5, 2.9))
            randomNum = random()
            
            try:
                if randomNum < 0.85:
                    if randomNum in [0.7, 0.1, 0.5, 0.21, 0.25]:
                        self.superLike()
                        likeCount += 1
                    else:
                        if self.like():
                            likeCount += 1
                        else:
                            self.handle_popups()
                else:
                    if self.dislike():
                        dislikeCount += 1
                    else:
                        self.handle_popups()

                print(f"Total Swipe: {likeCount + dislikeCount}")

            except Exception as e:
                print(f"Unexpected error occurred: {e}")

        print(
            f'Final Report ({likeCount+dislikeCount} swipes): \n Likes: {likeCount} \n Dislikes: {dislikeCount}')


bot = TinderBot()
bot.login()
# Wait facebook login redirect to main page
sleep(7)
bot.swipe()
