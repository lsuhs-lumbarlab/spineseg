import torch
import monai
import nibabel as nib

print("=" * 50)
print("ENVIRONMENT CHECK")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"MONAI version: {monai.__version__}")
print(f"Nibabel version: {nib.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")
print("=" * 50)
print("✓ All essential libraries imported successfully!")