# Roadmap: Lumbar Spine MRI Segmentation

A step-by-step guide to adapt this MONAI pipeline for production-ready lumbar spine segmentation.

---

## Phase 1: Data Acquisition & Preparation (Weeks 1-4)

### 1.1 Collect Lumbar Spine MRI Data
**Goal:** Gather 100-500 lumbar spine MRI scans

**Sources:**
- Hospital/clinic PACS systems
- Public datasets:
  - [Spider Dataset](https://spider.grand-challenge.org/) - Lumbar spine MRI
  - [VERSE Dataset](https://github.com/anjany/verse) - Vertebrae segmentation
  - [CT-Spine Dataset](https://github.com/ICT-MIRACLE-lab/CTPelvic1K) - CT but adaptable
  
**Requirements:**
- T1 or T2-weighted MRI sequences
- DICOM or NIfTI format
- IRB approval if using clinical data
- Anonymization (remove patient identifiers)

### 1.2 Create Segmentation Labels
**Goal:** Annotate structures in each scan

**What to segment:**
- Vertebral bodies (L1, L2, L3, L4, L5)
- Intervertebral discs (L1-L2, L2-L3, L3-L4, L4-L5, L5-S1)
- Spinal canal
- Optional: Spinous processes, facet joints

**Tools:**
- [3D Slicer](https://www.slicer.org/) - Free, powerful
- [ITK-SNAP](http://www.itksnap.org/) - Specialized for segmentation
- [MITK](https://www.mitk.org/) - Research platform

**Labeling strategy:**
```
Label values:
0  = Background
1  = L1 vertebra
2  = L2 vertebra
3  = L3 vertebra
4  = L4 vertebra
5  = L5 vertebra
6  = L1-L2 disc
7  = L2-L3 disc
8  = L3-L4 disc
9  = L4-L5 disc
10 = L5-S1 disc
11 = Spinal canal
```

**Time estimate:** 30-60 minutes per scan
**Tip:** Start with 20 high-quality annotations, train initial model, then use it to pre-segment remaining data (saves time!)

### 1.3 Data Organization
**Structure your data:**
```
data/
├── raw/
│   ├── images/
│   │   ├── patient001.nii.gz
│   │   ├── patient002.nii.gz
│   │   └── ...
│   └── labels/
│       ├── patient001_seg.nii.gz
│       ├── patient002_seg.nii.gz
│       └── ...
├── processed/
│   └── (preprocessed data will go here)
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

---

## Phase 2: Pipeline Adaptation (Weeks 5-6)

### 2.1 Update Transforms for Spine MRI

**Key differences from hippocampus:**
```python
# Spine MRI typically has:
# - Larger field of view (whole lumbar spine)
# - Anisotropic spacing (e.g., 0.5 x 0.5 x 3.0 mm)
# - More variation in patient positioning

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    
    # Resample to isotropic or consistent spacing
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.0, 1.0, 1.0),  # or (0.5, 0.5, 1.0) for finer resolution
        mode=("bilinear", "nearest")
    ),
    
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    
    # Crop to spine region (remove empty space)
    CropForegroundd(keys=["image", "label"], source_key="image"),
    
    # Normalize intensity (crucial for MRI)
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    
    # Larger patches for spine (vertebrae are bigger than hippocampus)
    RandSpatialCropd(
        keys=["image", "label"],
        roi_size=(96, 96, 96),  # or (128, 128, 128) if GPU allows
        random_size=False
    ),
    
    # Augmentation
    RandRotated(
        keys=["image", "label"],
        range_x=0.2,  # ~10 degrees
        range_y=0.2,
        range_z=0.2,
        prob=0.2,
        mode=("bilinear", "nearest")
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
])
```

### 2.2 Update Model Configuration

**Increase model capacity for more complex anatomy:**
```python
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=12,  # 11 structures + background
    channels=(32, 64, 128, 256, 512),  # Deeper network
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
)

# Or consider more advanced architectures:
# - SwinUNETR (better for capturing long-range spine structure)
# - SegResNet (efficient residual U-Net)
# - UNETR (transformer-based)
```

**Example with SwinUNETR:**
```python
from monai.networks.nets import SwinUNETR

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=12,
    feature_size=48,
    use_checkpoint=True,  # Saves GPU memory
)
```

### 2.3 Handle Class Imbalance

**Problem:** Vertebrae/discs are small compared to background

**Solutions:**
```python
# 1. Weighted loss
from monai.losses import DiceCELoss

loss_function = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    squared_pred=True,
    ce_weight=torch.tensor([0.1, 1.0, 1.0, 1.0, ...])  # Lower weight for background
)

# 2. Focal loss for hard examples
from monai.losses import DiceFocalLoss

loss_function = DiceFocalLoss(
    to_onehot_y=True,
    softmax=True,
    gamma=2.0,  # Focus on hard-to-segment regions
)

# 3. Combined approach
loss_dice = DiceLoss(to_onehot_y=True, softmax=True)
loss_focal = FocalLoss()
loss = loss_dice + 0.5 * loss_focal
```

---

## Phase 3: Training Strategy (Weeks 7-10)

### 3.1 Training Configuration
```python
# Hyperparameters for spine segmentation
config = {
    "batch_size": 2,  # Adjust based on GPU memory
    "learning_rate": 1e-4,
    "num_epochs": 200,  # Much longer than demo
    "validation_interval": 5,  # Validate every 5 epochs
    "warmup_epochs": 10,  # Gradual learning rate increase
}

# Learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=config["num_epochs"],
    eta_min=1e-6
)
```

### 3.2 Advanced Training Techniques

**A. Mixed Precision Training (faster, uses less memory)**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**B. Cache Dataset (faster data loading)**
```python
from monai.data import CacheDataset

train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,  # Cache 100% in RAM
    num_workers=4
)
```

**C. Early Stopping**
```python
best_metric = -1
patience = 20
patience_counter = 0

for epoch in range(num_epochs):
    # ... training ...
    
    if val_metric > best_metric:
        best_metric = val_metric
        patience_counter = 0
        save_checkpoint()
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### 3.3 Monitoring Training

**Use TensorBoard:**
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/spine_seg")

# During training:
writer.add_scalar("Loss/train", loss, epoch)
writer.add_scalar("Dice/val", dice_score, epoch)
writer.add_images("Predictions", pred_images, epoch)

# View in browser:
# tensorboard --logdir=runs
```

---

## Phase 4: Evaluation & Validation (Weeks 11-12)

### 4.1 Comprehensive Metrics
```python
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric
)

# Dice Score (overlap)
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

# Hausdorff Distance (boundary accuracy)
hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

# Average Surface Distance
asd_metric = SurfaceDistanceMetric(include_background=False)
```

### 4.2 Per-Structure Analysis
```python
# Evaluate each vertebra/disc separately
structure_names = ["L1", "L2", "L3", "L4", "L5", "L1-L2", ...]

for i, structure in enumerate(structure_names):
    dice_per_class = dice_metric.aggregate()[i]
    print(f"{structure}: Dice = {dice_per_class:.4f}")
```

### 4.3 Clinical Validation

**Work with radiologists to assess:**
- Vertebral level identification accuracy
- Disc boundary precision
- False positive/negative rates
- Qualitative assessment on test cases

---

## Phase 5: Post-Processing & Deployment (Weeks 13-14)

### 5.1 Post-Processing Pipeline
```python
from monai.transforms import (
    KeepLargestConnectedComponent,
    FillHoles,
    RemoveSmallObjects
)

post_process = Compose([
    # Remove small disconnected regions
    KeepLargestConnectedComponent(applied_labels=[1, 2, 3, 4, 5]),
    
    # Fill holes in vertebrae
    FillHoles(applied_labels=[1, 2, 3, 4, 5]),
    
    # Remove noise
    RemoveSmallObjects(min_size=100),
])

prediction = post_process(raw_prediction)
```

### 5.2 Vertebral Labeling Logic
```python
def assign_vertebral_levels(segmentation):
    """
    Assign L1-L5 labels based on spatial position
    (superior to inferior ordering)
    """
    # Find centroids of each connected component
    # Sort by z-coordinate (superior to inferior)
    # Assign labels L1 (top) to L5 (bottom)
    pass  # Implementation depends on your coordinate system
```

### 5.3 Integration Options

**A. Python API**
```python
# inference_api.py
class SpineSegmentationModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def segment(self, image_path):
        # Load, preprocess, predict, postprocess
        return segmentation_mask
```

**B. Command-line tool**
```bash
python segment_spine.py --input patient.nii.gz --output result.nii.gz
```

**C. Web service (Flask/FastAPI)**
```python
from fastapi import FastAPI, File

app = FastAPI()

@app.post("/segment")
async def segment_spine(file: UploadFile = File(...)):
    # Process uploaded MRI
    # Return segmentation
    pass
```

---

## Phase 6: Comparison with Existing Tools

### Study these existing spine segmentation tools:

**1. SpinePS (Spine Parsing and Segmentation)**
- Paper: [SpinePS on GitHub](https://github.com/Project-MONAI/tutorials)
- Uses: Multi-stage approach (detection → segmentation)
- Learn from: Architecture choices, label strategies

**2. TotalSpineSeg**
- Segments entire spine (cervical, thoracic, lumbar)
- Open-source implementation available
- Learn from: Data augmentation, post-processing

**3. Spinal Cord Toolbox (SCT)**
- Focus: Spinal cord segmentation (not vertebrae)
- Useful for: Cord/canal segmentation component
- Install: `pip install spinalcordtoolbox`
- Integration: Can use SCT for cord, your model for vertebrae

**4. nnU-Net (Medical Imaging Baseline)**
- Self-configuring framework
- Often state-of-the-art performance
- Consider using as baseline to compare against

---

## Critical Success Factors

### ✅ Data Quality
- **Most important!** Good annotations > fancy model
- Consistent labeling protocol
- Multiple annotators with consensus review

### ✅ Sufficient Data
- Minimum: 50 scans
- Good: 100-200 scans
- Excellent: 500+ scans
- Use data augmentation to artificially increase

### ✅ Proper Validation
- Hold out 15-20% for testing (never use during development)
- Cross-validation if data is limited
- Test on external dataset if possible

### ✅ Clinical Collaboration
- Work with radiologists throughout
- Validate clinical utility, not just metrics
- Consider edge cases (hardware, pathology, etc.)

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| 1. Data Collection | 4 weeks | Annotated dataset |
| 2. Pipeline Setup | 2 weeks | Adapted MONAI code |
| 3. Training | 4 weeks | Trained model |
| 4. Evaluation | 2 weeks | Performance metrics |
| 5. Deployment | 2 weeks | Inference system |
| **Total** | **14 weeks** | Production-ready model |

*Add buffer time for iterations and troubleshooting*

---

## Troubleshooting Common Issues

### GPU Out of Memory
```python
# Solutions:
1. Reduce batch_size
2. Reduce patch size (roi_size)
3. Use gradient checkpointing
4. Enable mixed precision training
5. Use model.eval() mode for inference
```

### Poor Segmentation Quality
```python
# Check:
1. Data quality (are labels accurate?)
2. Class imbalance (use weighted loss)
3. Training duration (try 2-3x more epochs)
4. Learning rate (try 1e-3, 1e-4, 1e-5)
5. Data augmentation (add more variety)
```

### Slow Training
```python
# Optimize:
1. Use CacheDataset or PersistentDataset
2. Increase num_workers in DataLoader
3. Use mixed precision training
4. Preprocess data offline
5. Use smaller validation set
```

---

## Resources & References

### Papers
- U-Net: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- nnU-Net: "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"
- UNETR: "UNETR: Transformers for 3D Medical Image Segmentation"

### Code Repositories
- MONAI Tutorials: https://github.com/Project-MONAI/tutorials
- MONAI Model Zoo: https://github.com/Project-MONAI/model-zoo
- Spine Segmentation Papers: https://paperswithcode.com/task/spine-segmentation

### Communities
- MONAI Forums: https://forums.projectmonai.io/
- Reddit: r/computervision, r/MachineLearning
- Discord: MONAI Discord server

---

## Next Immediate Actions

1. **This week**: Explore public spine datasets, download 10-20 samples
2. **Next week**: Set up 3D Slicer, practice manual segmentation
3. **Week 3-4**: Create labeling protocol, annotate first 20 scans
4. **Week 5**: Adapt current MONAI code for spine data
5. **Week 6**: Run first training experiment

---

**Good luck with your spine segmentation project! 🚀**

You now have all the foundational knowledge and a clear roadmap.