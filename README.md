# geofarmer-project
```
geofarmer/
│
├── README.md                         # Project overview, setup, pipeline, how to run
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Ignore large data / outputs / caches
├── config.py                         # Shared paths and project-wide settings
│
├── data/
│   ├── README.md                     # Explain where shared data lives and expected local structure
│   ├── sample/                       # Tiny sample only for testing code locally / in repo
│   └── processed/                    # Small generated csvs only if they are lightweight
│
├── notebooks/
│   ├── eda_geometry.ipynb            # Early exploration of masks / geometry features
│   ├── clustering_experiments.ipynb  # Raw baseline, skew/log transforms, PCA, k selection, cluster profiling
│   └── model_debug.ipynb             # Quick model testing / sanity checks
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_ftw.py               # Load metadata, images, masks, country subsets
│   │   ├── sample_data.py            # Country selection and chip sampling
│   │   └── build_metadata.py         # Create metadata tables with paths / ids
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── geometry_features.py      # Field-level geometry extraction from masks
│   │   ├── aggregate_chip_features.py# Aggregate field metrics to chip-level stats
│   │   └── image_stats.py            # RGB / NDVI / summary stats for baseline model
│   │
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── README.md                 # Notes on clustering experiments and scripts
│   │   └── cluster_chips.py          # Fit K-Means on geometry features and create cluster labels
│   │
│   ├── models/
│   │   ├── baseline/
│   │   │   ├── __init__.py
│   │   │   ├── model.py            # logistic regression / random forest
│   │   │   ├── train.py            # training + evaluation
│   │   │   └── features.py         # image stats (RGB, NDVI, etc.)
│   │   │
│   │   ├── cnn/
│   │   │   ├── __init__.py
│   │   │   ├── model.py            # CNN architecture
│   │   │   ├── train.py            # training loop
│   │   │   ├── dataset.py          # PyTorch dataset / dataloader
│   │   │   └── transforms.py       # preprocessing / augmentations
│   │   │
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Accuracy, F1, confusion matrix helpers
│   │   ├── compare_models.py         # Baseline vs CNN comparison
│   │   └── interpret_results.py      # Feature importance / result summaries
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py                     # Read/write csv, pickle, json helpers
│       ├── plotting.py               # Shared plotting functions
│       └── seed.py                   # Set random seeds for reproducibility
│
├── scripts/
│   ├── run_sampling.py               # Build selected subset
│   ├── run_geometry_features.py      # Generate chip-level geometry features
│   ├── run_image_stats.py            # Generate image-derived baseline inputs
│   ├── run_clustering.py             # Fit clustering + assign labels
│   ├── run_baseline.py               # Train/evaluate baseline model
│   ├── run_cnn.py                    # Train/evaluate CNN
│   └── run_full_pipeline.py          # Optional end-to-end runner
│
├── outputs/
│   ├── figures/                      # PCA plots, confusion matrices, example chips
│   ├── tables/                       # Metrics tables / cluster summaries
│   └── logs/                         # Training logs or saved run summaries
│
└── paper/
    ├── outline.md                    # Rough paper outline and section ownership
    ├── references.bib                # Citations
    ├── figures/                      # Final figures used in paper/slides
    └── draft/                        # Paper draft files 
```