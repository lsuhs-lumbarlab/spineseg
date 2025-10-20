import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm

print("="*50)
print("BUILDING SEGMENTATION MODEL")
print("="*50)

# Define the U-Net architecture
model = UNet(
    spatial_dims=3,          # 3D images
    in_channels=1,           # Grayscale input
    out_channels=3,          # 3 classes: background, anterior hippocampus, posterior hippocampus
    channels=(16, 32, 64, 128, 256),  # Number of filters at each level
    strides=(2, 2, 2, 2),    # Downsampling at each level
    num_res_units=2,         # Number of residual units
    norm=Norm.BATCH,         # Batch normalization
)

print("✓ Model created: 3D U-Net")
print(f"  Spatial dimensions: 3D")
print(f"  Input channels: 1")
print(f"  Output channels: 3 (background, class 1, class 2)")
print(f"  Architecture levels: 5")
print()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✓ Model parameters:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print()

# Test the model with a sample input
print("Testing model with sample input...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create a dummy input (batch_size=1, channels=1, H=32, W=32, D=32)
dummy_input = torch.randn(1, 1, 32, 32, 32).to(device)
print(f"  Input shape: {dummy_input.shape}")

# Forward pass
with torch.no_grad():
    output = model(dummy_input)

print(f"  Output shape: {output.shape}")
print(f"  Model device: {next(model.parameters()).device}")
print()

print("="*50)
print("✓ MODEL READY FOR TRAINING!")
print("="*50)