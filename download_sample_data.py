import os
from monai.apps import download_and_extract
import shutil

print("Downloading sample medical imaging data...")
print("This may take a few minutes...")

# Create a temporary directory for the download
resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
compressed_file = os.path.join("data", "raw", "Task04_Hippocampus.tar")
data_dir = os.path.join("data", "raw")

# Download and extract
download_and_extract(
    url=resource,
    output_dir=data_dir,
    hash_val=None,
)

print("\n" + "="*50)
print("✓ Sample data downloaded successfully!")
print(f"Location: {data_dir}")
print("="*50)