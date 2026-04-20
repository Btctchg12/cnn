import argparse
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

try:
    # Preferred import when called as part of the src.geometry package.
    from .chip_features import extract_chip_features
except ImportError:  # fallback for direct execution from this folder
    # Fallback import for local script-style execution from geometry/.
    from chip_features import extract_chip_features


def iter_chip_paths(input_dir, countries):
    """
    Yield (country, tif_path) for all chips under:
    input_dir/country/label_masks/instance/*.tif

    Parameters
    ----------
    input_dir : Path
        Root directory that contains one subfolder per country.
    countries : list[str] | None
        Optional list of country folder names. If None, all country folders
        found under ``input_dir`` are used.

    Yields
    ------
    tuple[str, Path]
        (country_name, path_to_chip_tif)
    """
    # Auto-discover country folders when the caller does not provide a list.
    if countries is None:
        countries = [p.name for p in input_dir.iterdir() if p.is_dir()]

    # Iterate country-by-country so downstream processing can be capped/logged.
    for country in countries:
        # Expected FTW layout for instance-mask chips.
        chip_dir = input_dir / country / "label_masks" / "instance"

        # Skip missing countries/folders gracefully.
        if not chip_dir.exists():
            print(f"[WARN] Missing chips folder for {country}: {chip_dir}")
            continue

        # Yield chips in sorted order for deterministic processing.
        for tif_path in sorted(chip_dir.glob("*.tif")):
            yield country, tif_path


def get_local_utm_crs(src):
    """Build a local UTM CRS from the raster centroid."""
    bounds = src.bounds
    lon = (bounds.left + bounds.right) / 2.0
    lat = (bounds.bottom + bounds.top) / 2.0
    zone = int(floor((lon + 180.0) / 6.0)) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def load_geometry_mask(src):
    """Load a mask and reproject it to a metric CRS when needed."""
    mask = src.read(1)

    if src.crs is not None and src.crs.is_geographic:
        dst_crs = get_local_utm_crs(src)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        reprojected = np.zeros((dst_height, dst_width), dtype=mask.dtype)
        reproject(
            source=mask,
            destination=reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

        pixel_area_sqm = float(abs(dst_transform.a * dst_transform.e))
        return reprojected, pixel_area_sqm

    pixel_area_sqm = float(abs(src.transform.a * src.transform.e))
    return mask, pixel_area_sqm


def process_dataset(input_dir, countries = None, max_chips_per_country = None,):
    """
    Extract chip-level geometry features for all selected chips.

    Parameters
    ----------
    input_dir : Path
        Root input directory containing country subfolders.
    countries : list[str] | None, default None
        Optional subset of countries to process.
    max_chips_per_country : int | None, default None
        Optional upper bound of processed chips per country.

    Returns
    -------
    pd.DataFrame
        One row per successfully processed chip with metadata columns
        (`chip_id`, `country`) followed by extracted feature columns.
    """
    # Collect one dictionary of features per processed chip.
    rows = []

    # Track per-country processed counts for optional capping.
    country_counts = {}

    # Iterate all target chip files discovered by folder traversal.
    for country, tif_path in iter_chip_paths(input_dir, countries):
        # Respect optional per-country cap for quicker/debug runs.
        if max_chips_per_country is not None:
            if country_counts.get(country, 0) >= max_chips_per_country:
                continue

        try:
            # Read the first raster band as the instance mask.
            with rasterio.open(tif_path) as src:
                mask, pixel_area_sqm = load_geometry_mask(src)

            # Convert one mask into one aggregated chip feature dictionary.
            features = extract_chip_features(mask, pixel_area_sqm=pixel_area_sqm)

            # No valid fields in this chip (all background or invalid fields).
            if features is None:
                print(f"[SKIP] No valid fields: {tif_path.name}")
                continue

            # Attach chip metadata used for grouping and traceability.
            features["chip_id"] = tif_path.stem
            features["country"] = country

            # Keep this chip row in the final dataset.
            rows.append(features)

            # Increment per-country processed count.
            country_counts[country] = country_counts.get(country, 0) + 1

        except Exception as e:
            # Continue processing other chips even if one chip fails.
            print(f"[ERROR] Failed on {tif_path}: {e}")

    # Build tabular output (empty if no chip succeeded).
    df = pd.DataFrame(rows)

    # Keep metadata columns first for readability in CSV output.
    if not df.empty:
        ordered_cols = ["chip_id", "country"] + [
            c for c in df.columns if c not in {"chip_id", "country"}
        ]
        df = df[ordered_cols]

    return df


def main() -> None:
    """CLI entrypoint for chip-level geometry feature extraction."""
    # Define command-line interface.
    parser = argparse.ArgumentParser(
        description="Extract chip-level geometry features from FTW instance masks."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root folder containing country folders.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Where to save the extracted CSV.",
    )
    parser.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help="Optional list of countries to process.",
    )
    parser.add_argument(
        "--max_chips_per_country",
        type=int,
        default=None,
        help="Optional cap for quick testing.",
    )
    args = parser.parse_args()

    # Run full extraction workflow.
    df = process_dataset(
        input_dir=args.input_dir,
        countries=args.countries,
        max_chips_per_country=args.max_chips_per_country,
    )

    # Report summary and exit early if nothing was produced.
    print(f"Processed {len(df)} chips.")
    if df.empty:
        print("[WARN] No output rows generated.")
        return

    # Ensure output folder exists before writing CSV.
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved features to {args.output_csv}")


if __name__ == "__main__":
    main()