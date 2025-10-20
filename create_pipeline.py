import os
import glob
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    RandRotate90d,
    RandFlipd,
    ToTensord,
)
from monai.data import Dataset, DataLoader, pad_list_data_collate
import torch

print("="*50)
print("CREATING MONAI DATA PIPELINE")
print("="*50)

# 1. Prepare data dictionary
data_root = os.path.join("data", "raw", "Task04_Hippocampus")
images_dir = os.path.join(data_root, "imagesTr")
labels_dir = os.path.join(data_root, "labelsTr")

image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

# Create list of dictionaries (MONAI's standard format)
data_dicts = [
    {"image": img, "label": lbl}
    for img, lbl in zip(image_files[:10], label_files[:10])  # Use first 10 for demo
]

print(f"✓ Prepared {len(data_dicts)} samples")
print(f"  Sample entry: {data_dicts[0]}")
print()

# 2. Define transforms
print("Setting up transforms...")

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),  # Load image and label
    EnsureChannelFirstd(keys=["image", "label"]),  # Ensure channel dimension exists
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.0, 1.0, 1.0),  # Resample to 1mm spacing
        mode=("bilinear", "nearest"),  # Bilinear for image, nearest for label
    ),
    Orientationd(keys=["image", "label"], axcodes="RAS"),  # Standard orientation
    ScaleIntensityRanged(
        keys=["image"],
        a_min=0.0,
        a_max=255.0,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),  # Normalize intensity to [0, 1]
    # Data augmentation (only for training)
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
])

print("✓ Transforms defined:")
print("  - LoadImaged: Load NIfTI files")
print("  - EnsureChannelFirstd: Add channel dimension")
print("  - Spacingd: Resample to consistent spacing")
print("  - Orientationd: Standardize orientation")
print("  - ScaleIntensityRanged: Normalize intensity")
print("  - RandRotate90d: Random 90° rotation (augmentation)")
print("  - RandFlipd: Random flip (augmentation)")
print()

# 3. Create dataset and dataloader
print("Creating dataset and dataloader...")

dataset = Dataset(data=data_dicts, transform=train_transforms)
dataloader = DataLoader(
    dataset, 
    batch_size=2, 
    shuffle=True, 
    num_workers=0,
    collate_fn=pad_list_data_collate  # Add this line
)

print(f"✓ Dataset created with {len(dataset)} samples")
print(f"✓ DataLoader created with batch_size=2")
print()

# 4. Test the pipeline by loading one batch
print("Testing pipeline - loading one batch...")

batch = next(iter(dataloader))
images = batch["image"]
labels = batch["label"]

print(f"✓ Batch loaded successfully!")
print(f"  Images shape: {images.shape}")  # Should be [batch_size, channels, H, W, D]
print(f"  Labels shape: {labels.shape}")
print(f"  Images device: {images.device}")
print(f"  Images dtype: {images.dtype}")
print(f"  Images value range: [{images.min():.3f}, {images.max():.3f}]")

print("\n" + "="*50)
print("✓ MONAI PIPELINE WORKING CORRECTLY!")
print("="*50)