"""Fit K-Means clustering model on all chip geometry features and output labeled data.

Example:
	python cluster_chips.py \
		--input-csv data/processed/geometry_metric_features_4115chips.csv \
		--output-csv data/labeled_geometry_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


GEO_FEATURES = [
	"log_num_fields",
	"log_mean_area_sqm",
	"area_cv",
	"coverage",
	"log_fields_per_covered_area",
	"mean_aspect_ratio",
	"largest_field_fraction",
	"mean_compactness",
	"mean_rectangularity",
	"std_rectangularity",
	"prop_rectangular_07",
	"mean_edge_density",
	"mean_cardinal_alignment",
]

BASE_REQUIRED_COLUMNS = ["num_fields", "mean_area_sqm", "fields_per_covered_area"]


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Add log-transformed features used by clustering."""
	out = df.copy()
	out["log_num_fields"] = np.log1p(out["num_fields"])
	out["log_mean_area_sqm"] = np.log1p(out["mean_area_sqm"])
	out["log_fields_per_covered_area"] = np.log1p(out["fields_per_covered_area"] * 1e6)
	return out


def validate_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
	missing = [c for c in columns if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns for {context}: {missing}")


def fit_pipeline(
	df: pd.DataFrame,
	n_clusters: int,
	pca_variance_threshold: float,
	random_state: int,
) -> tuple[pd.DataFrame, StandardScaler, PCA, KMeans, np.ndarray, np.ndarray]:
	validate_columns(df, BASE_REQUIRED_COLUMNS, "log feature creation")
	prepared = add_log_features(df)
	validate_columns(prepared, GEO_FEATURES, "model features")

	scaler = StandardScaler()
	x_std = scaler.fit_transform(prepared[GEO_FEATURES])

	pca_full = PCA(random_state=random_state)
	pca_full.fit(x_std)
	cum_var = np.cumsum(pca_full.explained_variance_ratio_)
	n_components = int(np.argmax(cum_var >= pca_variance_threshold) + 1)

	pca = PCA(n_components=n_components, random_state=random_state)
	x_pca = pca.fit_transform(x_std)

	kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
	labels = kmeans.fit_predict(x_pca)
	return prepared, scaler, pca, kmeans, x_pca, labels


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Fit K-Means clustering model on chip features")
	parser.add_argument(
		"--input-csv",
		default="data/processed/geometry_metric_features_4115chips.csv",
		help="Input CSV with geometry features",
	)
	parser.add_argument(
		"--output-csv",
		default="data/labeled_geometry_features.csv",
		help="Output CSV path for all data with predicted cluster labels",
	)
	parser.add_argument("--n-clusters", type=int, default=3, help="Number of K-Means clusters")
	parser.add_argument(
		"--pca-variance-threshold",
		type=float,
		default=0.90,
		help="Cumulative explained variance threshold used to choose PCA components",
	)
	parser.add_argument("--random-state", type=int, default=42)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	input_path = Path(args.input_csv)
	output_path = Path(args.output_csv)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(input_path)
	prepared, scaler, pca, kmeans, x_pca, labels = fit_pipeline(
		df=df,
		n_clusters=args.n_clusters,
		pca_variance_threshold=args.pca_variance_threshold,
		random_state=args.random_state,
	)

	labeled_df = prepared.copy()
	labeled_df["cluster"] = labels
	labeled_df.to_csv(output_path, index=False)

	print(f"Loaded rows: {len(df)}")
	print(f"PCA components selected: {pca.n_components_}")
	print(f"X_pca shape: {x_pca.shape}")
	print(f"Saved labeled CSV: {output_path.resolve()}")


if __name__ == "__main__":
	main()

	
