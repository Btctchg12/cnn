# Geometry Feature Extraction

## Core Purpose
- Convert instance-mask TIF chips into a metric-consistent chip-level geometry table (one row per chip) for country separation and clustering.

## What the pipeline does
This module reads instance-mask rasters, computes per-field geometry, aggregates those
measurements to the chip level, and writes a CSV with one row per chip.

The current implementation expects this layout:

`input_dir/<country>/label_masks/instance/*.tif`

Each raster should contain:
- band 1 as an instance mask
- `0` for background
- non-zero integers for field instance IDs

## Data Flow (End-to-End)
- Discover chips under `input_dir/country/label_masks/instance/*.tif`.
- Open each chip and read band 1 as an instance mask (`0` background, non-zero field IDs).
- If CRS is geographic, reproject mask to local UTM (centroid-based zone) with nearest-neighbor resampling.
- Compute `pixel_area_sqm` from raster transform.
- For each field ID, compute per-field geometry in `field_features.py`.
- Aggregate per-field outputs to chip-level statistics in `chip_features.py`.
- Attach `chip_id` and `country`.
- Write one CSV row per valid chip in `extract_features.py`.

## Files
- `field_features.py`: per-field geometry from one instance mask.
- `chip_features.py`: per-chip aggregation from per-field dictionaries.
- `extract_features.py`: dataset traversal, CRS handling, CSV output.
- `eda_geometry.ipynb`: visual/statistical analysis of extracted features.
- `scripts/run_geometry_features.py`: command-line wrapper for running the extraction pipeline from the repository root.

## Geometric Feature Dictionary

### Field-level (per instance)
- `area`: physical area in square meters (`area_px * pixel_area_sqm`).
- `area_px`: field area in pixels.
- `perimeter`: contour perimeter from `regionprops`.
- `compactness`: `4πA / P²` (clamped to `<= 1.0`).
- `rectangularity`: `area_px / axis_aligned_bbox_area_px`.
- `aspect_ratio`: `minor_axis_length / major_axis_length` (rotation-invariant).
- `edge_density`: `perimeter / area_px`.
- `cardinal_alignment`: `(major_axis_length * minor_axis_length) / axis_aligned_bbox_area_px`.

### Chip-level (aggregated output columns)
- `num_fields`: number of valid fields in chip.
- `mean_area`: mean field area (sqm).
- `mean_area_sqm`: explicit sqm alias of `mean_area`.
- `mean_area_px`: mean field area in pixels.
- `area_cv`: coefficient of variation of field area.
- `coverage`: total field area divided by chip area.
- `fields_per_covered_area`: field count normalized by total covered area.
- `mean_aspect_ratio`: mean field aspect ratio.
- `largest_field_fraction`: largest field area divided by total field area.
- `mean_compactness`: mean compactness.
- `mean_rectangularity`: mean rectangularity.
- `std_rectangularity`: rectangularity dispersion.
- `prop_rectangular_07`: fraction of fields with rectangularity `> 0.7`.
- `mean_edge_density`: mean edge density.
- `mean_cardinal_alignment`: mean cardinal alignment.
- `median_cardinal_alignment`: median cardinal alignment.

| Output column | Meaning | Units |
| --- | --- | --- |
| `chip_id` | Chip filename stem used as the chip identifier | none |
| `country` | Country folder name for the chip | none |
| `num_fields` | Number of valid field instances in the chip | count |
| `mean_area` | Mean field area | $m^2$ |
| `mean_area_sqm` | Same value as `mean_area`, kept as an explicit metric-area alias | $m^2$ |
| `mean_area_px` | Mean field area | pixels |
| `area_cv` | Coefficient of variation of field area | none |
| `coverage` | Fraction of chip area covered by labeled fields | fraction |
| `fields_per_covered_area` | Field count normalized by total covered area | fields / $m^2$ |
| `mean_aspect_ratio` | Mean intrinsic aspect ratio (`minor_axis_length / major_axis_length`) | none |
| `largest_field_fraction` | Largest field area divided by total field area | fraction |
| `mean_compactness` | Mean compactness across fields | none |
| `mean_rectangularity` | Mean rectangularity across fields | fraction |
| `std_rectangularity` | Standard deviation of rectangularity across fields | none |
| `prop_rectangular_07` | Fraction of fields with rectangularity above 0.7 | fraction |
| `mean_edge_density` | Mean perimeter-to-area ratio across fields | $m^{-1}$ |
| `mean_cardinal_alignment` | Mean ratio of intrinsic box area to axis-aligned box area | none |
| `median_cardinal_alignment` | Median ratio of intrinsic box area to axis-aligned box area | none |

## Critical Architectural Decisions
- Reproject to local UTM before area/perimeter math so geometric quantities are in metric units and comparable across countries.
- Use `skimage.measure.regionprops` for intrinsic shape descriptors to reduce orientation bias from axis-aligned-only approximations.

## How to run the pipeline

### 1) Install dependencies
From the project root:

```bash
pip install -r requirements.txt
```

### 2) Run geometry extraction
The simplest way to generate the chip-level CSV is:

```bash
python scripts/run_geometry_features.py \
	--input_dir data/sample_data \
	--output_csv outputs/geometry_metric_features.csv
```

To limit processing to specific countries:

```bash
python scripts/run_geometry_features.py \
	--input_dir data/sample_data \
	--countries france rwanda \
	--output_csv outputs/geometry_metric_features.csv
```

For a quick smoke test, cap the number of chips per country:

```bash
python scripts/run_geometry_features.py \
	--input_dir data/sample_data \
	--max_chips_per_country 1 \
	--output_csv outputs/geometry_metric_features.csv
```