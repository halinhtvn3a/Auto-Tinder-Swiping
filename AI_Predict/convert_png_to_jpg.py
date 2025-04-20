"""
Script để chuyển đổi các file PNG sang JPG trong thư mục dislike
"""
import os
import sys
from PIL import Image
from pathlib import Path

def convert_png_to_jpg(input_folder):
    """
    Chuyển đổi tất cả file PNG sang JPG trong thư mục chỉ định
    
    Args:
        input_folder: Thư mục chứa các file cần chuyển đổi
    """
    # Đảm bảo thư mục tồn tại
    if not os.path.exists(input_folder):
        print(f"Thư mục {input_folder} không tồn tại.")
        return
        
    # Đếm số lượng file đã chuyển đổi
    converted_count = 0
    
    # Tìm tất cả file PNG trong thư mục
    png_files = list(Path(input_folder).glob('*.png'))
    
    print(f"Tìm thấy {len(png_files)} file PNG trong thư mục {input_folder}")
    
    for png_file in png_files:
        try:
            # Đọc file PNG
            img = Image.open(png_file)
            
            # Tạo tên file JPG mới
            jpg_file = png_file.with_suffix('.jpg')
            
            # Lưu ảnh với định dạng JPG
            img.convert('RGB').save(jpg_file, "JPEG", quality=95)
            
            # Xóa file PNG gốc sau khi chuyển đổi thành công
            os.remove(png_file)
            
            converted_count += 1
            print(f"Đã chuyển đổi: {png_file.name} -> {jpg_file.name}")
            
        except Exception as e:
            print(f"Lỗi khi chuyển đổi {png_file.name}: {str(e)}")
    
    print(f"\nHoàn tất! Đã chuyển đổi {converted_count} file từ PNG sang JPG.")

if __name__ == "__main__":
    # Thư mục chứa ảnh dislike
    dislike_folder = os.path.join("PreferenceRecognition", "data", "dislike")
    
    # Thực hiện chuyển đổi
    convert_png_to_jpg(dislike_folder)