#!/usr/bin/env python
"""
Script kiểm tra CUDA và GPU với PyTorch
Sử dụng: python check_pytorch_gpu.py
"""

import os
import sys
import subprocess
import time

def install_pytorch():
    """Cài đặt PyTorch nếu chưa có"""
    print("Đang kiểm tra PyTorch...")
    try:
        import torch
        print(f"PyTorch đã được cài đặt, phiên bản: {torch.__version__}")
        return True
    except ImportError:
        print("PyTorch chưa được cài đặt. Đang cài đặt...")
        try:
            # Cài đặt phiên bản PyTorch có CUDA
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
            print("Đã cài đặt PyTorch thành công!")
            return True
        except Exception as e:
            print(f"Không thể cài đặt PyTorch: {e}")
            return False

def check_gpu_with_pytorch():
    """Kiểm tra GPU với PyTorch"""
    try:
        import torch
        
        print("\n===== KIỂM TRA GPU VỚI PYTORCH =====")
        
        # Kiểm tra CUDA có sẵn không
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print("✅ CUDA khả dụng với PyTorch!")
            print(f"- Số lượng GPU: {torch.cuda.device_count()}")
            
            # Liệt kê các GPU
            for i in range(torch.cuda.device_count()):
                print(f"- GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  - CUDA Capability: {torch.cuda.get_device_capability(i)}")
                print(f"  - Bộ nhớ toàn bộ: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # Kiểm tra hiệu suất GPU
            print("\nĐang chạy kiểm tra hiệu suất...")
            
            # Tạo tensor trên GPU và CPU
            size = 5000
            
            # Làm nóng GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                x_gpu = torch.randn(size, size, device='cuda')
                y_gpu = torch.randn(size, size, device='cuda')
                torch.cuda.synchronize()
            
            # Benchmark CPU
            start_time = time.time()
            x_cpu = torch.randn(size, size)
            y_cpu = torch.randn(size, size)
            z_cpu = torch.matmul(x_cpu, y_cpu)
            cpu_time = time.time() - start_time
            
            # Benchmark GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_time = time.time()
                z_gpu = torch.matmul(x_gpu, y_gpu)
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time
                
                # So sánh hiệu suất
                speedup = cpu_time / gpu_time
                print(f"Thời gian CPU: {cpu_time:.4f} giây")
                print(f"Thời gian GPU: {gpu_time:.4f} giây")
                print(f"Tăng tốc: {speedup:.2f}x")
                
                if speedup > 5:
                    print("✅ GPU hoạt động hiệu quả!")
                elif speedup > 1:
                    print("⚠️ GPU hoạt động nhưng không nhanh như kỳ vọng")
                else:
                    print("⚠️ GPU không nhanh hơn CPU!")
                
                # Kiểm tra vị trí lưu trữ tensor
                print(f"\nTensor được lưu trên: {z_gpu.device}")
                
        else:
            print("❌ CUDA không khả dụng với PyTorch")
            print("\nNguyên nhân có thể:")
            print("1. Không có GPU NVIDIA")
            print("2. Driver NVIDIA chưa được cài đặt")
            print("3. CUDA Toolkit chưa được cài đặt")
            print("4. Phiên bản PyTorch không hỗ trợ CUDA")

        # Thông tin cài đặt CUDA của PyTorch
        print(f"\nThông tin PyTorch CUDA:")
        if hasattr(torch.version, 'cuda'):
            print(f"- Phiên bản CUDA: {torch.version.cuda}")
        if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'version'):
            print(f"- Phiên bản cuDNN: {torch.backends.cudnn.version()}")
        
        # Kết luận
        print("\n===== KẾT LUẬN =====")
        if cuda_available:
            print("PyTorch nhận diện được GPU của bạn.")
            print("Nếu TensorFlow không sử dụng GPU, vấn đề có thể nằm ở:")
            print("1. Phiên bản TensorFlow không tương thích với CUDA")
            print("2. Thư viện CUDA cho TensorFlow bị thiếu")
            print("3. Cấu hình TensorFlow không đúng")
            print("\nKhuyến nghị: Thử sử dụng Docker với image TensorFlow GPU")
        else:
            print("PyTorch không nhận diện GPU. Vấn đề có thể nằm ở driver hoặc CUDA.")
            print("Cần cài đặt hoặc cập nhật driver NVIDIA và CUDA Toolkit.")
        
    except Exception as e:
        print(f"Lỗi khi kiểm tra GPU với PyTorch: {e}")

def check_system_nvidia():
    """Kiểm tra thông tin hệ thống và driver NVIDIA"""
    print("\n===== KIỂM TRA DRIVER NVIDIA =====")
    
    try:
        output = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
        print("✅ Driver NVIDIA đã được cài đặt:")
        print(output)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Không thể chạy nvidia-smi, driver NVIDIA có thể chưa được cài đặt hoặc lỗi")
        print("Khuyến nghị: Tải và cài đặt driver NVIDIA mới nhất từ https://www.nvidia.com/Download/index.aspx")

def main():
    """Hàm chính"""
    print("===== CÔNG CỤ KIỂM TRA GPU VỚI PYTORCH =====")
    print("Tool này sẽ kiểm tra xem PyTorch có nhận diện được GPU của bạn hay không")
    print("Điều này giúp xác định liệu vấn đề nằm ở TensorFlow hay ở cấu hình GPU")

    # Cài đặt PyTorch nếu cần
    if install_pytorch():
        # Kiểm tra driver NVIDIA
        check_system_nvidia()
        
        # Kiểm tra GPU với PyTorch
        check_gpu_with_pytorch()

if __name__ == "__main__":
    main()