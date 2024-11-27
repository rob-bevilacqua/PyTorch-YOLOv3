import torch
import torchvision

# Test NMS with dummy data
boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15], [10, 10, 20, 20]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)

try:
    indices = torchvision.ops.nms(boxes, scores, 0.5)
    print(f"NMS indices: {indices}")
except Exception as e:
    print(f"NMS failed: {e}")

