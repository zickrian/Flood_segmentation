# Flood Segmentation using Deep Learning

A comprehensive deep learning project for flood area segmentation using semantic segmentation models. This project implements and compares two state-of-the-art architectures: **U-Net** and **DeepLabv3+** for identifying flooded regions in aerial/satellite imagery.

## 🌊 Project Overview

This project tackles the critical problem of flood detection and segmentation using computer vision and deep learning. By training neural networks on the FloodNet dataset, the models learn to accurately identify and segment flooded areas in images, which can be crucial for disaster response and management.

### Key Features

- **Dual Model Architecture**: Implementation of both U-Net and DeepLabv3+ models
- **Comprehensive Data Pipeline**: Automated data loading, preprocessing, and augmentation
- **Advanced Metrics**: Evaluation using IoU (Intersection over Union), Dice Coefficient, and Pixel Accuracy
- **Threshold Optimization**: Automatic threshold tuning on validation set for optimal performance
- **Visualization**: Clear visualization of predictions with overlay comparisons
- **GPU Acceleration**: Optimized for GPU training with mixed precision support

## 📊 Dataset

The project uses the **FloodNet** dataset from Kaggle:
- **Source**: [FloodNet Challenge @ EARTHVISION 2021](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation)
- **Content**: High-resolution aerial images of flood-affected areas
- **Structure**:
  - Training images: `train-org-img/`
  - Training masks: `train-label-img/`
  - Test images: `test-org-img/`
  - Test masks: `test-label-img/`

### Dataset Statistics
- **Image Size**: 256x256 pixels (resized)
- **Channels**: RGB (3 channels) for images, Binary (1 channel) for masks
- **Split Ratio**: 70% Training, 15% Validation, 15% Testing
- **Total Samples**: Variable based on downloaded dataset

## 🏗️ Model Architectures

### 1. U-Net
A classic encoder-decoder architecture for biomedical image segmentation:
- **Encoder**: Convolutional blocks with max-pooling for feature extraction
- **Decoder**: Upsampling with skip connections from encoder
- **Skip Connections**: Preserves spatial information across scales
- **Output**: Single channel sigmoid activation for binary segmentation

**Architecture Details**:
```
Input (256x256x3) → Encoder Blocks → Bottleneck → Decoder Blocks → Output (256x256x1)
Filter sizes: 64 → 128 → 256 → 512 → 1024 (encoder)
```

### 2. DeepLabv3+
Advanced architecture with atrous spatial pyramid pooling:
- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **ASPP Module**: Multi-scale feature extraction with atrous convolution
- **Decoder**: Low-level feature fusion for refined boundaries
- **Advantages**: Better handling of objects at multiple scales

**Architecture Details**:
```
Input (256x256x3) → MobileNetV2 Backbone → ASPP → Decoder → Output (256x256x1)
Atrous rates: [6, 12, 18]
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Kaggle API credentials

### Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn kaggle opencv-python
pip install keras-unet-collection
```

### Setup Kaggle API

1. Create a Kaggle account and generate API token
2. Upload `kaggle.json` to your environment:
```python
from google.colab import files
files.upload()  # Upload kaggle.json
```

3. Configure Kaggle:
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 🚀 Usage

### 1. Data Preparation

```python
# Download dataset from Kaggle
!kaggle datasets download -d faizalkarim/flood-area-segmentation
!unzip -q flood-area-segmentation.zip
```

### 2. Data Loading

The project includes custom functions for loading and preprocessing:
```python
def read_image(path, size=(256, 256)):
    """Load and normalize RGB image"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

def read_mask(path, size=(256, 256)):
    """Load and process binary mask"""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, size)
    mask = (mask > 127).astype(np.uint8)
    return mask[..., np.newaxis]
```

### 3. Model Training

#### Train U-Net:
```python
# Build model
unet_model = build_unet(input_shape=(256, 256, 3))

# Configure callbacks
callbacks = [
    ModelCheckpoint('unet_best.keras', save_best_only=True),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5)
]

# Train
history = unet_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks
)
```

#### Train DeepLabv3+:
```python
# Build model
deeplabv3_model = build_deeplabv3plus(input_shape=(256, 256, 3))

# Train with same callback strategy
history = deeplabv3_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks
)
```

### 4. Evaluation

The project includes comprehensive evaluation metrics:
```python
def iou_dice_acc(pred, true):
    """Calculate IoU, Dice, and Pixel Accuracy"""
    intersection = np.sum(pred * true)
    union = np.sum(pred) + np.sum(true) - intersection
    
    iou = intersection / (union + 1e-7)
    dice = (2 * intersection) / (np.sum(pred) + np.sum(true) + 1e-7)
    pixel_acc = np.sum(pred == true) / pred.size
    
    return iou, dice, pixel_acc
```

### 5. Threshold Optimization

Automatic threshold tuning on validation set:
```python
def find_best_threshold(model, X_val, y_val, thresholds=np.linspace(0.30, 0.70, 9)):
    """Find optimal prediction threshold"""
    # Tests multiple thresholds and returns best based on IoU
```

### 6. Prediction

```python
def predict_mask(image, model, threshold=0.5):
    """Generate segmentation mask for new image"""
    logit = model.predict(image[None, ...], verbose=0)[0, ..., 0]
    prob = 1 / (1 + np.exp(-logit))
    pred = (prob > threshold).astype(np.uint8)
    return prob, pred
```

## 📈 Results

### Model Performance Comparison

#### U-Net Performance:
```
Metric              | Value
--------------------|-------
Mean IoU            | ~0.75
Mean Dice Score     | ~0.86
Mean Pixel Accuracy | ~0.90
```

#### DeepLabv3+ Performance:
```
Metric              | Value
--------------------|-------
Mean IoU            | ~0.73
Mean Dice Score     | ~0.84
Mean Pixel Accuracy | ~0.87
```

### Key Findings:
- Both models achieve strong performance on flood segmentation
- U-Net shows slightly better results in this implementation
- DeepLabv3+ excels at handling multi-scale features
- Threshold optimization improves final IoU by ~2-5%

### Sample Results

The notebook includes visualizations showing:
1. **Original Image**: Input aerial/satellite imagery
2. **Ground Truth Mask**: Human-annotated flood regions
3. **Predicted Mask**: Model-generated segmentation
4. **Overlay**: Visual comparison with colored overlay

## 🗂️ Project Structure

```
Flood_segmentation/
│
├── Flood_segmentation.ipynb    # Main Jupyter notebook
│   ├── Import Libraries
│   ├── Data Loading & Preprocessing
│   ├── Data Exploration
│   ├── Data Augmentation
│   ├── U-Net Implementation
│   ├── U-Net Training & Evaluation
│   ├── DeepLabv3+ Implementation
│   ├── DeepLabv3+ Training & Evaluation
│   └── Threshold Optimization & Final Results
│
└── README.md                   # This file
```

## 🛠️ Technical Details

### Data Augmentation
The project implements various augmentation techniques:
- Horizontal/Vertical flipping
- Rotation (90°, 180°, 270°)
- Brightness adjustment
- Contrast adjustment
- Gaussian noise

### Training Configuration
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Binary Cross-Entropy + Dice Loss
- **Batch Size**: Configurable (typically 16-32)
- **Epochs**: 50 with early stopping
- **Learning Rate**: 1e-4 with ReduceLROnPlateau
- **Mixed Precision**: Enabled for GPU optimization

### Custom Losses
```python
def dice_loss(y_true, y_pred):
    """Dice loss for better boundary detection"""
    smooth = 1e-7
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def combined_loss(y_true, y_pred):
    """Combines BCE and Dice loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice
```

## 🎯 Use Cases

This flood segmentation system can be applied to:
- **Disaster Response**: Quick assessment of flood-affected areas
- **Emergency Planning**: Identifying vulnerable regions
- **Insurance Assessment**: Damage estimation and claims processing
- **Urban Planning**: Flood risk analysis and mitigation
- **Environmental Monitoring**: Long-term flood pattern analysis
- **Research**: Climate change impact studies

## 🔍 Evaluation Metrics Explained

### IoU (Intersection over Union)
- Measures overlap between predicted and true flood regions
- Range: 0 to 1 (higher is better)
- Also known as Jaccard Index

### Dice Coefficient
- Similar to IoU but more sensitive to small changes
- Range: 0 to 1 (higher is better)
- Better for imbalanced datasets

### Pixel Accuracy
- Percentage of correctly classified pixels
- Range: 0 to 1 (higher is better)
- Can be misleading for imbalanced classes

## 🚀 Future Improvements

Potential enhancements for the project:
- [ ] Add more advanced architectures (U-Net++, SegFormer)
- [ ] Implement ensemble methods
- [ ] Add temporal analysis for video sequences
- [ ] Deploy as web application
- [ ] Add multi-class segmentation (different flood severity levels)
- [ ] Implement real-time inference
- [ ] Add explainability visualizations (GradCAM)
- [ ] Support for higher resolution images
- [ ] Transfer learning from other disaster datasets

## 📝 Notes

- The notebook is designed to run on Google Colab with GPU acceleration
- Dataset download requires Kaggle API credentials
- Training time varies based on GPU availability (typically 1-2 hours per model)
- Models are saved with `.keras` extension for compatibility

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Additional model architectures
- Enhanced data augmentation strategies
- Improved visualization tools
- Documentation improvements
- Bug fixes and optimizations

## 📄 License

This project is created for educational and research purposes. Please refer to the FloodNet dataset license for data usage terms.

## 🙏 Acknowledgments

- **FloodNet Dataset**: EARTHVISION 2021 Workshop Challenge
- **Kaggle**: For hosting the dataset
- **TensorFlow/Keras**: Deep learning framework
- **U-Net Paper**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **DeepLabv3+ Paper**: Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This project demonstrates practical application of semantic segmentation for disaster management. The models and techniques can be adapted for other segmentation tasks in computer vision.
