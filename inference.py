import os
import glob
import torch
import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference

print("="*70)
print("RUNNING INFERENCE ON NEW DATA")
print("="*70)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load trained model
print("Step 1: Loading trained model...")
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
print("✓ Model loaded successfully\n")

# Prepare test data
print("Step 2: Preparing test data...")
data_root = os.path.join("data", "raw", "Task04_Hippocampus")
images_dir = os.path.join(data_root, "imagesTr")

image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))

# Use images 10-15 (unseen during training/validation)
test_files = [{"image": img} for img in image_files[10:15]]
print(f"✓ Selected {len(test_files)} test images (indices 10-14)")
for i, f in enumerate(test_files):
    print(f"  {i+1}. {os.path.basename(f['image'])}")
print()

# Define transforms
print("Step 3: Setting up transforms...")
test_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
])

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

print("✓ Transforms configured\n")

# Run inference
print("Step 4: Running inference...")
print("-"*70)

output_dir = os.path.join("outputs", "predictions")
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        test_inputs = test_data["image"].to(device)
        
        # Run sliding window inference
        test_outputs = sliding_window_inference(
            test_inputs, 
            roi_size=(32, 32, 32), 
            sw_batch_size=1, 
            predictor=model
        )
        
        # Convert to segmentation labels
        prediction = torch.argmax(test_outputs, dim=1).squeeze().cpu().numpy()
        
        # Get original filename
        original_file = test_files[i]["image"]
        filename = os.path.basename(original_file).replace('.nii.gz', '_seg.nii.gz')
        
        # Load original image to get header info
        original_img = nib.load(original_file)
        
        # Save prediction as NIfTI with same header
        prediction_img = nib.Nifti1Image(prediction, original_img.affine, original_img.header)
        output_path = os.path.join(output_dir, filename)
        nib.save(prediction_img, output_path)
        
        print(f"✓ Processed: {os.path.basename(original_file)}")
        print(f"  Input shape: {test_inputs.shape}")
        print(f"  Prediction shape: {prediction.shape}")
        print(f"  Predicted classes: {np.unique(prediction)}")
        print(f"  Saved to: {filename}\n")

print("="*70)
print("INFERENCE COMPLETE!")
print(f"Predictions saved to: {output_dir}/")
print("="*70)
print("\nYou can now:")
print("1. Load these .nii.gz files in medical imaging viewers (ITK-SNAP, 3D Slicer)")
print("2. Compare predictions with ground truth")
print("3. Use them for further analysis")