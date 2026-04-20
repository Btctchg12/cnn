# CNN Classification Pipeline

## Core Purpose
- Train and evaluate a lightweight CNN for classifying farmland-structure cluster labels from Sentinel-2 satellite image chips.
- Use paired multi-temporal image inputs (`window_a` and `window_b`) and predict the `cluster` label stored in the split CSV files.

## What the pipeline does
This module builds an end-to-end CNN workflow for satellite image classification.

The current implementation:
- reads labeled chip metadata from CSV files,
- loads `.tif` imagery from `window_a` and `window_b`,
- concatenates the two windows into a single multi-channel tensor,
- normalizes pixel values,
- trains a CNN classifier with PyTorch,
- saves trained model weights,
- evaluates predictions on a held-out test set using standard classification metrics.

## Data Flow (End-to-End)
- Read `train.csv` or `test.csv` containing `chip_id`, `country`, and `cluster`.
- For each row, locate corresponding image files under each country's `s2_images/window_a/` and `s2_images/window_b/` directories.
- Load each `.tif` image as a float32 tensor.
- Concatenate `window_a` and `window_b` along the channel dimension.
- Replace NaN / inf values with `0.0`.
- Normalize pixel values by dividing by `10000.0`.
- Feed the resulting tensor into the CNN model.
- During training:
  - compute cross-entropy loss,
  - update parameters with Adam,
  - track training loss and accuracy.
- During evaluation:
  - generate class predictions,
  - compute Accuracy, Macro F1, Confusion Matrix, and Classification Report.

## Expected Directory Layout
The current code expects image files to follow this structure:

`parent/<country>/s2_images/window_a/<chip_id>.tif`  
`parent/<country>/s2_images/window_b/<chip_id>.tif`

It also expects split CSV files such as:

`data/split/train.csv`  
`data/split/test.csv`

The dataset loader resolves file paths from `chip_id` and `country`, and skips samples whose required image files are missing. :contentReference[oaicite:5]{index=5}

## Files
- `cnn_dataset.py`: custom PyTorch `Dataset` for loading and preprocessing multi-window Sentinel-2 `.tif` chips. :contentReference[oaicite:6]{index=6}
- `cnn_model.py`: CNN architecture definition (`SimpleCNN`). :contentReference[oaicite:7]{index=7}
- `cnn_train.py`: training loop, optimizer/loss setup, and model checkpoint saving. :contentReference[oaicite:8]{index=8}
- `cnn_evaluation.py`: test-time inference and metric reporting. :contentReference[oaicite:9]{index=9}
- `cnn_split.py`: utility for generating a train/test split from the labeled geometry CSV. :contentReference[oaicite:10]{index=10}
- `simple_cnn_ab.pth`: saved trained model weights used for evaluation.

## Dataset Behavior

### Input requirements
The dataset class requires:
- a CSV file with columns:
  - `chip_id`
  - `country`
  - label column (default: `cluster`)
- at least one of:
  - `window_a`
  - `window_b`

If both are enabled, the images are concatenated channel-wise into one tensor. If neither is enabled, the loader raises an error. The loader also supports optional country filtering and optional metadata return. :contentReference[oaicite:11]{index=11}

### Output
Each dataset sample returns:
- `image`: PyTorch tensor of shape `[C, H, W]`
- `label`: integer class label

If `return_metadata=True`, it additionally returns:
- `chip_id`
- `country`

## Model Architecture

### CNN structure
The current model is a compact 3-block convolutional classifier:

1. `Conv2d(in_channels, 16, kernel_size=3, padding=1)`  
2. `ReLU`  
3. `MaxPool2d(2)`  

4. `Conv2d(16, 32, kernel_size=3, padding=1)`  
5. `ReLU`  
6. `MaxPool2d(2)`  

7. `Conv2d(32, 64, kernel_size=3, padding=1)`  
8. `ReLU`  
9. `AdaptiveAvgPool2d((1, 1))`  

Classifier head:
- `Flatten`
- `Linear(64, 64)`
- `ReLU`
- `Dropout(0.3)`
- `Linear(64, num_classes)`

This design keeps the model lightweight while still extracting hierarchical spatial features from the satellite image chips. :contentReference[oaicite:12]{index=12}

### Input / Output
- Input channels: `8`
- Output classes: `3`

The code sets `in_channels=8` because it uses both `window_a` and `window_b` together during training and evaluation. :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}

## Train/Test Split
The split utility reads `data/labeled_geometry_features.csv`, keeps only:
- `chip_id`
- `country`
- `cluster`

and creates:
- `90%` training data
- `10%` test data

using:
- `random_state=42`
- shuffled sampling

The outputs are saved to:
- `data/split/train.csv`
- `data/split/test.csv` :contentReference[oaicite:15]{index=15}

## Training Procedure
The training script:
- loads the training split from `data/split/train.csv`,
- builds a `DataLoader` with `batch_size=4` and `shuffle=True`,
- instantiates `SimpleCNN(in_channels=8, num_classes=3)`,
- uses:
  - `CrossEntropyLoss`
  - `Adam(lr=1e-3)`
- trains for `10` epochs,
- prints batch progress and epoch-level training loss / accuracy,
- saves the trained model to:

`src/models/cnn/simple_cnn_ab.pth` :contentReference[oaicite:16]{index=16}

## Evaluation Procedure
The evaluation script:
- loads the test split from `data/split/test.csv`,
- rebuilds the same dataset and model configuration,
- loads model weights from `simple_cnn_ab.pth`,
- runs inference without gradient computation,
- collects predictions across the full test set,
- reports:
  - Accuracy
  - Macro F1
  - Confusion Matrix
  - Classification Report :contentReference[oaicite:17]{index=17}

## Critical Architectural Decisions
- Use both `window_a` and `window_b` as model input to incorporate multi-temporal image information. 
- Normalize pixel values by dividing by `10000.0`, which is standard for scaled Sentinel-style reflectance inputs. :contentReference[oaicite:19]{index=19}
- Use a lightweight CNN instead of a deeper architecture to keep training simple and computationally manageable. :contentReference[oaicite:20]{index=20}
- Use adaptive average pooling before the classifier to reduce the parameter count compared with a large flattened feature map. :contentReference[oaicite:21]{index=21}

## How to run the pipeline

### 1) Install dependencies
From the project root:

```bash
pip install torch pandas numpy rasterio scikit-learn
