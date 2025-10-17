import os
import glob
import nibabel as nib
import numpy as np
from monai.transforms import LoadImage

# Find the downloaded data
data_root = os.path.join("data", "raw", "Task04_Hippocampus")
images_dir = os.path.join(data_root, "imagesTr")
labels_dir = os.path.join(data_root, "labelsTr")

# Get list of all image files
image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

print("="*50)
print("DATA EXPLORATION")
print("="*50)
print(f"Number of training images: {len(image_files)}")
print(f"Number of training labels: {len(label_files)}")
print()

# Load first image and label using nibabel
if len(image_files) > 0:
    print("Loading first sample with nibabel...")
    img_path = image_files[0]
    label_path = label_files[0]
    
    img = nib.load(img_path)
    label = nib.load(label_path)
    
    print(f"\nImage path: {os.path.basename(img_path)}")
    print(f"Image shape: {img.shape}")
    print(f"Image data type: {img.get_data_dtype()}")
    print(f"Voxel spacing: {img.header.get_zooms()}")
    
    print(f"\nLabel shape: {label.shape}")
    print(f"Unique label values: {np.unique(label.get_fdata())}")
    
    print("\n" + "-"*50)
    print("Loading same image with MONAI LoadImage...")
    
    # Load with MONAI
    loader = LoadImage(image_only=True)
    monai_img = loader(img_path)
    
    print(f"MONAI image shape: {monai_img.shape}")
    print(f"MONAI image type: {type(monai_img)}")
    
    print("\n" + "="*50)
    print("✓ Data exploration complete!")
    print("="*50)
else:
    print("ERROR: No image files found. Check the download.")