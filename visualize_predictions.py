import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, ScaleIntensityRanged
)
from monai.data import Dataset
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
import glob

print("="*70)
print("VISUALIZING MODEL PREDICTIONS")
print("="*70)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load the trained model
print("Loading trained model...")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

model.load_state_dict(torch.load(os.path.join("models", "best_model.pth")))
model.eval()
print("✓ Model loaded\n")

# Prepare one validation sample
data_root = os.path.join("data", "raw", "Task04_Hippocampus")
images_dir = os.path.join(data_root, "imagesTr")
labels_dir = os.path.join(data_root, "labelsTr")

image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

# Use validation sample
test_data = [{"image": image_files[8], "label": label_files[8]}]

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
])

test_ds = Dataset(data=test_data, transform=transforms)
test_item = test_ds[0]

# Run inference
print("Running inference...")
with torch.no_grad():
    test_input = test_item["image"].unsqueeze(0).to(device)
    test_output = sliding_window_inference(test_input, (32, 32, 32), 1, model)
    test_prediction = torch.argmax(test_output, dim=1).squeeze().cpu().numpy()

test_image = test_item["image"].squeeze().cpu().numpy()
test_label = test_item["label"].squeeze().cpu().numpy()

print(f"✓ Inference complete")
print(f"  Image shape: {test_image.shape}")
print(f"  Prediction unique values: {np.unique(test_prediction)}")
print(f"  Ground truth unique values: {np.unique(test_label)}\n")

# Visualize middle slice
slice_idx = test_image.shape[2] // 2

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(test_image[:, :, slice_idx], cmap="gray")
plt.title("Input MRI (Middle Slice)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(test_image[:, :, slice_idx], cmap="gray")
plt.imshow(test_label[:, :, slice_idx], cmap="jet", alpha=0.5, vmin=0, vmax=2)
plt.title("Ground Truth Segmentation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(test_image[:, :, slice_idx], cmap="gray")
plt.imshow(test_prediction[:, :, slice_idx], cmap="jet", alpha=0.5, vmin=0, vmax=2)
plt.title("Model Prediction")
plt.axis("off")

plt.tight_layout()
output_path = os.path.join("outputs", "prediction_visualization.png")
os.makedirs("outputs", exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")
print("\n" + "="*70)

plt.show()