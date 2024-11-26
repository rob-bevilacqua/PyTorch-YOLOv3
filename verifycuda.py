import torch
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("PyTorch version:", torch.__version__)
print("CUDA version (PyTorch):", torch.version.cuda)

# Create a tensor and move it to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.rand(10000, 10000).to(device)

print(f"Tensor is on device: {tensor.device}")