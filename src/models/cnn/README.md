# CNN 

## Core Purpose
- Train a CNN model to classify farmland structure types using Sentinel-2 satellite imagery.
- Use paired multi-temporal inputs (`window_a` and `window_b`) to predict cluster labels and compare with the cluster label from `labeled_geometry_features.csv`.
## Inputs
- A parent folder containing all the files.
- order: parent-> country ->s2_images-> window_a and window_b
- files can be downloaded from: https://data.source.coop/kerner-lab/fields-of-the-world-archive/rwanda.zip ( may need to change rwanda to other countries' names)

## What the pipeline does
The current implementation:
- Reads chip metadata (chip_id, country...) from `labeled_geometry_features.csv` and uses them to locate image files.
- extracts `.tif` imagery from `window_a` and `window_b`,
- concatenates the two windows into a single multi-channel tensor,
- trains a CNN classifier, 90% of data for training, 10% for testing
- saves trained mode,
- evaluates predictions on a held-out test set using accuracy, F1 score, and confusion matrix.

## Files
- `cnn_dataset.py`: load and preprocess multi-window Sentinel-2 `.tif` chips.
- `cnn_model.py`: CNN architecture definition.
- `cnn_train.py`: training loop, optimizer/loss setup, and model checkpoint saving. 
- `cnn_evaluation.py`: comparing the label from the CNN model with the label in the `labeled_geometry_features.csv`.
- `cnn_split.py`: generate train/test split CSV.

## How to run the pipeline

### 1) Install dependencies
From the project root:

```bash
pip install torch pandas numpy rasterio scikit-learn
