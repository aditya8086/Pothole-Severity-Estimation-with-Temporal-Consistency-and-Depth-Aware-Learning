import torch
from models.yolo import SegmentationModel

# Use safe_globals as a CONTEXT MANAGER (required in PyTorch 2.6+)
with torch.serialization.safe_globals([SegmentationModel]):
    ckpt = torch.load(
        "Pothole_model.pt",
        map_location="cpu",
        weights_only=False
    )

model = ckpt["model"]

print("Number of classes (nc):", model.nc)
print("Class names stored in model:", getattr(model, "names", None))
