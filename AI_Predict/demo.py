import os
import sys
import time
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import module
root_dir = str(Path(__file__).parent.absolute())
sys.path.append(root_dir)

from PreferenceRecognition import (
    PreferenceModel,
    DataCollector,
    ImageProcessor,
    TinderClient,
    PreferenceRecognizer
)

def demo_data_collection():
    """Demo thu thập dữ liệu và phân loại ảnh"""
    print("==== DEMO THU THẬP DỮ LIỆU ====")
    
    collector = DataCollector()
    stats = collector.get_dataset_stats()
    print(f"Thống kê dữ liệu hiện tại: {stats}")
    
    # Nếu có thư mục ảnh để phân loại
    test_images_dir = os.path.join(root_dir, "test_images")
    if os.path.exists(test_images_dir):
        print(f"Phân loại ảnh từ {test_images_dir}")
        
        # Giả lập phân loại ảnh: thích các file có tên chứa "like"
        for file in os.listdir(test_images_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                is_like = 'like' in file.lower()
                filepath = os.path.join(test_images_dir, file)
                saved_path = collector.save_image(filepath, is_like)
                print(f"Phân loại ảnh {file} là {'LIKE' if is_like else 'DISLIKE'} -> {saved_path}")

def demo_model_training():
    """Demo huấn luyện mô hình"""
    print("\n==== DEMO HUẤN LUYỆN MÔ HÌNH ====")
    
    model = PreferenceModel()
    collector = DataCollector()
    stats = collector.get_dataset_stats()
    
    # Kiểm tra xem có đủ dữ liệu để huấn luyện không
    if stats['like_count'] > 10 and stats['dislike_count'] > 10:
        print(f"Huấn luyện mô hình với {stats['total_count']} mẫu")
        model.build_model(transfer_learning=True)
        history = model.train(
            training_dir=collector.data_dir,
            epochs=5,  # Chỉ chạy 5 epochs cho demo
            batch_size=16
        )
        print("Huấn luyện mô hình hoàn tất!")
    else:
        print(f"Không đủ dữ liệu để huấn luyện. Cần ít nhất 10 mẫu mỗi loại.")
        print(f"Hiện có: Like={stats['like_count']}, Dislike={stats['dislike_count']}")

def demo_image_prediction():
    """Demo dự đoán sở thích trên ảnh"""
    print("\n==== DEMO DỰ ĐOÁN SỞ THÍCH ====")
    
    model = PreferenceModel()
    processor = ImageProcessor()
    
    # Thử tải mô hình đã huấn luyện
    if model.load_trained_model():
        print("Đã tải thành công mô hình đã huấn luyện!")
        
        # Tìm một số ảnh để dự đoán
        test_images_dir = os.path.join(root_dir, "test_images")
        if os.path.exists(test_images_dir):
            for file in os.listdir(test_images_dir)[:5]:  # Chỉ dự đoán 5 ảnh đầu tiên
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(test_images_dir, file)
                    print(f"Dự đoán ảnh: {file}")
                    
                    # Tiền xử lý ảnh
                    img = processor.preprocess_image(filepath)
                    
                    # Dự đoán
                    score = model.predict(img)
                    decision = "LIKE" if score > 0.7 else "DISLIKE"
                    
                    print(f"  -> Kết quả: {decision} (điểm số: {score:.4f})")
                    
                    time.sleep(1)  # Tạm dừng để dễ theo dõi
    else:
        print("Không tìm thấy mô hình đã huấn luyện.")
        print("Hãy chạy demo_model_training() trước.")

def demo_tinder_api():
    """Demo tương tác với Tinder API"""
    print("\n==== DEMO TƯƠNG TÁC TINDER API ====")
    
    # Lưu ý: Để demo này hoạt động, người dùng cần có Tinder token
    token = input("Nhập Tinder X-Auth-Token (để trống để bỏ qua demo này): ")
    
    if not token:
        print("Bỏ qua demo Tinder API.")
        return
        
    client = TinderClient(auth_token=token)
    
    # Lấy thông tin profile của bản thân
    my_profile = client.get_self_profile()
    if my_profile:
        print(f"Đã kết nối với Tinder API. Xin chào {my_profile.get('name', 'User')}!")
        
        # Lấy danh sách suggestions
        print("Đang lấy danh sách người dùng được đề xuất...")
        recs = client.get_recs()
        
        if recs:
            print(f"Đã tìm thấy {len(recs)} suggestions.")
            
            # Hiển thị thông tin một vài profile
            for i, profile in enumerate(recs[:3]):  # Chỉ hiển thị 3 profile đầu tiên
                user_id = profile.get('_id')
                name = profile.get('name', 'Unknown')
                bio = profile.get('bio', 'No bio')
                images = client.extract_profile_images(profile)
                
                print(f"\nProfile {i+1}: {name} (ID: {user_id})")
                print(f"Bio: {bio[:100]}{'...' if len(bio) > 100 else ''}")
                print(f"Số ảnh: {len(images)}")
                
                if images:
                    print(f"Ảnh đầu tiên: {images[0]}")
        else:
            print("Không thể lấy danh sách người dùng được đề xuất.")
    else:
        print("Không thể kết nối với Tinder API. Hãy kiểm tra token.")

def demo_full_system():
    """Demo toàn bộ hệ thống"""
    print("\n==== DEMO TOÀN BỘ HỆ THỐNG ====")
    
    # Lưu ý: Để demo này hoạt động, người dùng cần có Tinder token
    token = input("Nhập Tinder X-Auth-Token (để trống để chạy chế độ mô phỏng): ")
    
    # Khởi tạo hệ thống
    system = PreferenceRecognizer(tinder_token=token)
    
    # Cấu hình
    system.set_config({
        'threshold': 0.65,  # Điều chỉnh ngưỡng để like
        'max_profiles_per_day': 5,  # Giới hạn 5 profiles cho demo
        'delay_between_swipes': 3,  # Delay 3 giây giữa các lần swipe
        'collect_data': True  # Thu thập dữ liệu mới
    })
    
    # Nếu có mô hình đã huấn luyện, sử dụng nó
    try:
        model_loaded = system.model.load_trained_model()
        if model_loaded:
            print("Đã tải thành công mô hình đã huấn luyện!")
        else:
            print("Không tìm thấy mô hình đã huấn luyện. Sử dụng chiến lược ngẫu nhiên.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
    
    # Bắt đầu tự động quẹt
    print("\nBắt đầu tự động quẹt...")
    if not token:
        print("Chạy chế độ mô phỏng...")
        # TODO: Thêm code mô phỏng ở đây nếu không có token
        print("Kết thúc mô phỏng.")
    else:
        try:
            # Giới hạn chỉ quẹt tối đa 5 profiles và chạy trong 2 phút
            stats = system.auto_swipe(count=5, duration_minutes=2)
            print("\nKết quả:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        except KeyboardInterrupt:
            print("\nĐã dừng quẹt theo yêu cầu người dùng.")
        except Exception as e:
            print(f"\nLỗi khi quẹt: {e}")

if __name__ == "__main__":
    print("DEMO HỆ THỐNG NHẬN DIỆN SỞ THÍCH TINDER")
    print("=======================================")
    
    while True:
        print("\nChọn demo để chạy:")
        print("1. Thu thập và phân loại dữ liệu")
        print("2. Huấn luyện mô hình")
        print("3. Dự đoán sở thích trên ảnh")
        print("4. Tương tác với Tinder API")
        print("5. Chạy toàn bộ hệ thống")
        print("0. Thoát")
        
        choice = input("\nLựa chọn của bạn: ")
        
        if choice == '1':
            demo_data_collection()
        elif choice == '2':
            demo_model_training()
        elif choice == '3':
            demo_image_prediction()
        elif choice == '4':
            demo_tinder_api()
        elif choice == '5':
            demo_full_system()
        elif choice == '0':
            print("Kết thúc demo.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
            
        input("\nNhấn Enter để tiếp tục...")