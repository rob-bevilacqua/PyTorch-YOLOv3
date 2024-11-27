import torch
print("Is CUDA available?:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
