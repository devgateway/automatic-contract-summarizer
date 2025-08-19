import torch

def test_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    test_cuda()