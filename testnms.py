import torch
import torchvision

# Dummy data
boxes = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float32).to("cuda")
scores = torch.tensor([0.9, 0.8], dtype=torch.float32).to("cuda")
iou_threshold = 0.5

try:
    kept_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    print("NMS ran successfully on CUDA!")
except Exception as e:
    print("NMS failed:", e)
