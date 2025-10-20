# Spine Segmentation with MONAI

A complete end-to-end deep learning pipeline for medical image segmentation using MONAI.

## Project Structure
```
spineseg/
├── data/
│   ├── raw/              # Original medical images
│   └── processed/        # Preprocessed data
├── models/
│   └── best_model.pth    # Trained model checkpoint
├── outputs/
│   ├── predictions/      # Segmentation predictions
│   └── prediction_visualization.png
├── scripts/              # Utility scripts
├── check_setup.py        # Verify environment setup
├── explore_data.py       # Data exploration
├── create_pipeline.py    # MONAI data pipeline demo
├── build_model.py        # Model architecture
├── train.py              # Training script
├── inference.py          # Run predictions on new data
├── visualize_predictions.py  # Visualize results
└── README.md
```

## Setup

### Environment
- Python with Conda
- PyTorch 2.6.0 + CUDA 12.4
- MONAI 1.5.1
- GPU: NVIDIA RTX 4070 SUPER

### Installation
```bash
pip install monai[all]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Key MONAI Concepts Learned

### 1. Data Pipeline
- **Transforms**: Preprocessing and augmentation pipeline
  - `LoadImaged`: Load NIfTI files
  - `Spacingd`: Resample to consistent voxel spacing
  - `Orientationd`: Standardize orientation (RAS)
  - `ScaleIntensityRanged`: Normalize intensity
  - `RandSpatialCropd`, `RandRotate90d`, `RandFlipd`: Data augmentation

- **Dataset & DataLoader**: 
  - Dictionary format: `{"image": path, "label": path}`
  - `pad_list_data_collate`: Handle variable-sized images in batches

### 2. Model Architecture
- **3D U-Net**: Standard for medical image segmentation
  - Input: 1 channel (grayscale MRI)
  - Output: 3 channels (multi-class segmentation)
  - ~4.8M parameters

### 3. Training
- **Loss Function**: `DiceCELoss` (Dice + Cross Entropy)
- **Metric**: Dice Score (0-1, higher is better)
- **Optimizer**: Adam with learning rate 1e-4
- **Epochs**: Number of complete passes through training data

### 4. Inference
- **Sliding Window Inference**: Process large 3D volumes in patches
- **Output**: Segmentation masks as NIfTI files

## Usage

### Check Setup
```bash
python check_setup.py
```

### Explore Data
```bash
python explore_data.py
```

### Train Model
```bash
python train.py
```

### Run Inference
```bash
python inference.py
```

### Visualize Results
```bash
python visualize_predictions.py
```

## Training Results

- **Training samples**: 8
- **Validation samples**: 2
- **Epochs**: 5
- **Best Dice Score**: 0.0596
- **Final Loss**: 2.0264

*Note: Low performance is expected with only 8 training samples and 5 epochs. Real projects use 100+ samples and 50-200 epochs.*

## Next Steps for Spine Segmentation

1. **Collect spine MRI data** with segmentation labels
2. **Increase dataset size** (aim for 100+ scans)
3. **Adjust architecture** for spine-specific features
4. **Train longer** (100-200 epochs)
5. **Tune hyperparameters** (learning rate, batch size, loss weights)
6. **Implement vertebral labeling** (L1-L5 identification)
7. **Add post-processing** (connected components, morphological operations)

## Medical Image Formats

- **NIfTI (.nii.gz)**: Standard 3D medical imaging format
- **Voxel spacing**: Physical distance between voxels (e.g., 1mm × 1mm × 1mm)
- **Orientation**: Standard is RAS (Right-Anterior-Superior)

## Resources

- [MONAI Documentation](https://docs.monai.io/)
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- [SpineSeg Papers](https://github.com/topics/spine-segmentation)

## License

Research and educational use.