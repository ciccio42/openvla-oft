import tensorflow as tf
import torch
import os
def test_gpu():
    print("Testing PyTorch GPU availability...")
    if torch.cuda.is_available():
        print(f"PyTorch GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No PyTorch GPU available")
        return False
    
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU found")
        return False
    
    
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Tensorflow GPUs found: {gpus}")
        
        
        return True
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
        return False
    
    
    
    
if __name__ == "__main__":
    # print LD_LIBRARY_PATH and PATH
    print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', 'Not set'))
    print("PATH:", os.environ.get('PATH', 'Not set'))
    
    if test_gpu():
        print("GPU is available and configured correctly.")
    else:
        print("GPU configuration failed or no GPU available.")