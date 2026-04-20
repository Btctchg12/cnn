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
```

### 2) Download all the necessary files from the website and store them in a parent folder

### 3) Run these files in order. `cnn_dataset.py`,`cnn_model.py`,`cnn_split.py`,`cnn_train.py`,`cnn_evaluation.py`.

## Evaluation Results

## 📊 Results

```text
===== TEST RESULT =====
Accuracy: 0.6456
Macro F1: 0.6016

Confusion Matrix:
[[112   8  57]
 [ 11  23  29]
 [ 36   5 131]]

Classification Report:
              precision    recall  f1-score   support

           0     0.7044    0.6328    0.6667       177
           1     0.6389    0.3651    0.4646        63
           2     0.6037    0.7616    0.6735       172

    accuracy                         0.6456       412
   macro avg     0.6490    0.5865    0.6016       412
weighted avg     0.6523    0.6456    0.6386       412
```

