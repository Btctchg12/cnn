# Clustering Module

This folder contains clustering code for assigning geometry-based cluster labels to chips.

## Script

- `cluster_chips.py`: Fits K-Means on selected geometry features after log transforms, standardization, and PCA.

## Expected Input

- Default input CSV: `data/processed/geometry_metric_features_4115chips.csv`
- Required base columns for log feature creation:
	- `num_fields`
	- `mean_area_sqm`
	- `fields_per_covered_area`
- Also expects all model features used in the script (`GEO_FEATURES`).

## Output

- Default output CSV: `data/labeled_geometry_features.csv`
- Output contains the prepared input rows plus a new `cluster` column.

## Run

From the project root:

```bash
python src/clustering/cluster_chips.py
```

With explicit arguments:

```bash
python src/clustering/cluster_chips.py \
	--input-csv data/processed/geometry_metric_features_4115chips.csv \
	--output-csv data/labeled_geometry_features.csv \
	--n-clusters 3 \
	--pca-variance-threshold 0.90 \
	--random-state 42
```
