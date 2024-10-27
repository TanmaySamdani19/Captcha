import torch

def check_gpu_availability():
    # Check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available!")
        # Get GPU details
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("GPU is not available.")

# Run the function
check_gpu_availability()
