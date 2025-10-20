import os
import glob
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, RandRotate90d, RandFlipd, RandSpatialCropd,
)
from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
import numpy as np

print("="*70)
print("MONAI SEGMENTATION TRAINING")
print("="*70)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print()

# 1. PREPARE DATA
print("Step 1: Preparing data...")
data_root = os.path.join("data", "raw", "Task04_Hippocampus")
images_dir = os.path.join(data_root, "imagesTr")
labels_dir = os.path.join(data_root, "labelsTr")

image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

# Split into train/validation
train_files = [{"image": img, "label": lbl} for img, lbl in zip(image_files[:8], label_files[:8])]
val_files = [{"image": img, "label": lbl} for img, lbl in zip(image_files[8:10], label_files[8:10])]

print(f"  Training samples: {len(train_files)}")
print(f"  Validation samples: {len(val_files)}")
print()

# 2. DEFINE TRANSFORMS
print("Step 2: Setting up transforms...")
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    RandSpatialCropd(keys=["image", "label"], roi_size=(32, 32, 32), random_size=False),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
])

print("  ✓ Training transforms: with augmentation")
print("  ✓ Validation transforms: without augmentation")
print()

# 3. CREATE DATASETS AND DATALOADERS
print("Step 3: Creating datasets...")
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=pad_list_data_collate)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=pad_list_data_collate)

print(f"  ✓ Train DataLoader: {len(train_loader)} batches")
print(f"  ✓ Validation DataLoader: {len(val_loader)} batches")
print()

# 4. CREATE MODEL
print("Step 4: Building model...")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

print(f"  ✓ Model: 3D U-Net with {sum(p.numel() for p in model.parameters()):,} parameters")
print()

# 5. DEFINE LOSS AND OPTIMIZER
print("Step 5: Setting up loss and optimizer...")
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

print("  ✓ Loss: Dice + Cross Entropy")
print("  ✓ Optimizer: Adam (lr=1e-4)")
print("  ✓ Metric: Dice Score")
print()

# 6. TRAINING LOOP
print("="*70)
print("Starting Training...")
print("="*70)

num_epochs = 5
best_metric = -1

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 70)
    
    # Training
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        print(f"  Train Step {step}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    epoch_loss /= step
    print(f"  → Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            
            # Use sliding window inference for larger images
            val_outputs = sliding_window_inference(val_inputs, (32, 32, 32), 1, model)
            val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
            
            dice_metric(y_pred=val_outputs, y=val_labels)
        
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        
        print(f"  → Validation Dice Score: {metric:.4f}")
        
        # Save best model
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), os.path.join("models", "best_model.pth"))
            print(f"  → ✓ New best model saved! (Dice: {best_metric:.4f})")

print("\n" + "="*70)
print(f"Training Complete! Best Dice Score: {best_metric:.4f}")
print(f"Model saved to: models/best_model.pth")
print("="*70)