import numpy as np

try:
    # Preferred import when used as part of the src.geometry package.
    from .field_features import compute_field_features
except ImportError:  # fallback for direct execution from this folder
    # Fallback import for local script-style execution from geometry/.
    from field_features import compute_field_features


def extract_chip_features(mask, pixel_area_sqm=1.0):
    """
    Compute chip-level aggregated geometry features from an instance mask.

    Parameters
    ----------
    mask : np.ndarray
        2D instance mask where:
        - 0 denotes background pixels
        - each non-zero integer denotes one field instance ID
    pixel_area_sqm : float, default 1.0
        Area represented by one pixel, in square meters when the source raster
        has been reprojected into a metric CRS.

    Returns
    -------
    dict | None
        Dictionary of chip-level summary features computed across all valid
        fields in the chip. Returns None when no non-background IDs exist or
        when no valid per-field feature rows can be produced.

    Notes
    -----
    This function aggregates per-field geometry produced by
    `compute_field_features` into one summary row per chip.

    The returned feature dictionary contains metric-area features when
    `pixel_area_sqm` is meaningful, which is why `mean_area` and
    `mean_area_sqm` are identical in this module.
    """
    # Collect all unique IDs in the mask and drop the background ID 0.
    field_ids = np.unique(mask)
    field_ids = field_ids[field_ids != 0]

    # No non-background IDs means no fields in this chip.
    if len(field_ids) == 0:
        return None

    # Compute per-field geometry for each instance ID.
    field_features = []
    for field_id in field_ids:
        # Isolate one field and compute its geometry features.
        feats = compute_field_features(mask, int(field_id), pixel_area_sqm=pixel_area_sqm)

        # Skip IDs that produce invalid/empty geometry outputs.
        if feats is not None:
            field_features.append(feats)

    # If every ID failed per-field extraction, stop here.
    if not field_features:
        return None

    # Convert each per-field metric into NumPy arrays for vectorized summaries.
    areas = np.asarray([f["area"] for f in field_features], dtype=float)
    areas_px = np.asarray([f["area_px"] for f in field_features], dtype=float)
    compactnesses = np.asarray([f["compactness"] for f in field_features], dtype=float)
    rectangularities = np.asarray([f["rectangularity"] for f in field_features], dtype=float)
    aspect_ratios = np.asarray([f["aspect_ratio"] for f in field_features], dtype=float)
    edge_densities = np.asarray([f["edge_density"] for f in field_features], dtype=float)
    cardinal_alignments = np.asarray([f["cardinal_alignment"] for f in field_features], dtype=float)

    # Total labeled field area and full chip area (all pixels in the array).
    total_field_area = float(np.sum(areas))
    chip_area = float(mask.size * pixel_area_sqm)

    # Coverage = fraction of chip occupied by labeled fields.
    coverage = total_field_area / (chip_area + 1e-6)

    mean_area = float(np.mean(areas))

    chip_features = {
        # Number of valid field instances in the chip.
        "num_fields": len(field_features),

        # Mean field size in square meters.
        "mean_area": mean_area,
        # Explicit alias kept for readability in downstream analysis.
        "mean_area_sqm": mean_area,
        # Mean field size in pixels.
        "mean_area_px": float(np.mean(areas_px)),

        # Coefficient of variation of area: std/mean, captures size variability.
        "area_cv": float(np.std(areas) / (np.mean(areas) + 1e-6)),

        # Share of chip pixels covered by fields.
        "coverage": float(coverage),

        # Field count normalized by covered area (density-like measure).
        "fields_per_covered_area": float(len(field_features) / (total_field_area + 1e-6)),

        # Average width/height ratio across fields.
        "mean_aspect_ratio": float(np.mean(aspect_ratios)),

        # Fraction of total field area contributed by the largest field.
        "largest_field_fraction": float(np.max(areas) / (total_field_area + 1e-6)),

        # Mean shape compactness across fields.
        "mean_compactness": float(np.mean(compactnesses)),

        # Mean area/bounding-box-area ratio across fields.
        "mean_rectangularity": float(np.mean(rectangularities)),

        # Spread of rectangularity values across fields.
        "std_rectangularity": float(np.std(rectangularities)),

        # Proportion of fields with rectangularity above a heuristic threshold.
        "prop_rectangular_07": float(np.mean(rectangularities > 0.7)),

        # Mean perimeter-to-area ratio across fields.
        "mean_edge_density": float(np.mean(edge_densities)),

        # Orientation summary; lower values indicate more rotation away from
        # the cardinal axes.
        "mean_cardinal_alignment": float(np.mean(cardinal_alignments)),
        "median_cardinal_alignment": float(np.median(cardinal_alignments)),
    }

    # One aggregated feature row representing this entire chip.
    return chip_features