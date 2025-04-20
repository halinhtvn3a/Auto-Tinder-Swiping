"""
Script kiểm tra cài đặt TensorFlow và GPU
"""
import os
import sys
import traceback

print("===== KIỂM TRA CÀI ĐẶT TENSORFLOW =====")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print()

# Kiểm tra các biến môi trường liên quan đến CUDA
print("CUDA Environment Variables:")
for env_var in ['CUDA_VISIBLE_DEVICES', 'TF_FORCE_GPU_ALLOW_GROWTH', 
                'TF_XLA_FLAGS', 'TF_GPU_THREAD_MODE', 'LD_LIBRARY_PATH', 
                'CUDA_PATH']:
    print(f"{env_var}: {os.environ.get(env_var, 'Not set')}")
print()

# Kiểm tra import tensorflow
print("Checking TensorFlow import...")
try:
    import tensorflow
    print(f"TensorFlow module path: {tensorflow.__file__}")
    print(f"TensorFlow version (from module): {getattr(tensorflow, '__version__', 'Not available')}")
    
    # Kiểm tra chi tiết về phiên bản
    try:
        import tensorflow.version
        print(f"TensorFlow version from version module: {tensorflow.version.VERSION}")
    except (ImportError, AttributeError) as e:
        print(f"Could not get version from version module: {e}")
    
    # Kiểm tra GPU
    print("\nGPU Configuration:")
    try:
        print(f"Built with CUDA: {tensorflow.test.is_built_with_cuda()}")
        physical_devices = tensorflow.config.list_physical_devices('GPU')
        print(f"GPU devices detected: {len(physical_devices)}")
        print(f"GPU devices: {physical_devices}")
        
        if len(physical_devices) > 0:
            print("\nGPU Device Details:")
            for i, device in enumerate(physical_devices):
                print(f"  Device {i}: {device}")
                
            # Kiểm tra thử một phép tính trên GPU
            try:
                print("\nTesting GPU computation...")
                with tensorflow.device('/GPU:0'):
                    a = tensorflow.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tensorflow.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tensorflow.matmul(a, b)
                    print(f"Tensor result: {c.numpy()}")
                    print(f"Computed on device: {c.device}")
            except Exception as e:
                print(f"Error during GPU computation test: {e}")
    except Exception as e:
        print(f"Error checking GPU configuration: {e}")
        traceback.print_exc()
        
except ImportError as e:
    print(f"Failed to import TensorFlow: {e}")
    print("\nInstalled packages:")
    try:
        import pkg_resources
        packages = sorted([f"{p.key}=={p.version}" for p in pkg_resources.working_set])
        for package in packages[:20]:  # Showing only first 20 to avoid clutter
            print(f"  {package}")
        if len(packages) > 20:
            print(f"  ... and {len(packages)-20} more")
    except ImportError:
        print("Could not list installed packages")

print("\n===== KIỂM TRA HOÀN TẤT =====")