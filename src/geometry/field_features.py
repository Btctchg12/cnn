import numpy as np
from skimage.measure import regionprops

def compute_bbox_dims(field_mask):
    """
    Return tight bounding-box height and width for a binary field mask.

    Parameters
    ----------
    field_mask : np.ndarray
        2D binary-like mask for a single field.

    Returns
    -------
    tuple[int, int]
        (height, width). Returns (0, 0) when the mask has no foreground.
    """
    # Guard against unexpected shapes/types to avoid axis errors.
    if not isinstance(field_mask, np.ndarray):
        raise TypeError("field_mask must be a NumPy array")
    if field_mask.ndim != 2:
        raise ValueError("field_mask must be a 2D array")

    # Convert to bool so "occupied" means foreground pixel present.
    field_mask = field_mask.astype(bool, copy=False)

    # Row/column occupancy vectors: True where at least one field pixel exists.
    rows = np.any(field_mask, axis=1)
    cols = np.any(field_mask, axis=0)

    # Indices of occupied rows/cols define the tight bounding range.
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]

    # Empty foreground => no valid box.
    if len(row_idx) == 0 or len(col_idx) == 0:
        return 0, 0

    # Inclusive span (+1) from first to last occupied index.
    height = int(row_idx[-1] - row_idx[0] + 1)
    width = int(col_idx[-1] - col_idx[0] + 1)
    return height, width


def compute_field_features(mask, field_id, pixel_area_sqm=1.0):
    """
    Compute geometry features for a single field ID inside an instance mask.

    Parameters
    ----------
    mask : np.ndarray
        2D instance mask where 0 is background and each non-zero integer value
        represents one field instance ID.
    field_id : int
        Field instance ID to isolate and summarize.

    Returns
    -------
    dict | None
        Feature dictionary for the selected field, or None if the field does
        not exist / has zero area.
    """
    # Validate mask shape/type before any array operations.
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy array")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    # Build a binary mask for exactly one field instance.
    field_mask = (mask == field_id)

    # Area is the count of foreground pixels.
    area_px = int(np.sum(field_mask))

    # Missing field ID in this chip.
    if area_px == 0:
        return None

    # Compute intrinsic shape statistics from skimage.regionprops.
    props = regionprops(field_mask.astype(np.uint8))
    if not props:
        return None
    prop = props[0]

    # Preserve metric area downstream by converting pixel area to square meters.
    area_sqm = float(area_px * pixel_area_sqm)

    # Intrinsic perimeter from regionprops is more accurate than grid XOR.
    perimeter = float(prop.perimeter)
    if perimeter <= 0:
        return None

    # Axis-aligned bounding box dimensions for legacy rectangularity.
    height, width = compute_bbox_dims(field_mask)
    if height == 0 or width == 0:
        return None

    axis_bbox_area_px = max(height * width, 1)

    major_axis_length = float(getattr(prop, "axis_major_length", getattr(prop, "major_axis_length", 0.0)))
    minor_axis_length = float(getattr(prop, "axis_minor_length", getattr(prop, "minor_axis_length", 0.0)))
    if major_axis_length <= 0 or minor_axis_length <= 0:
        return None

    intrinsic_bbox_area_px = max(major_axis_length * minor_axis_length, 1e-6)

    # Compactness: 4πA/P² (closer to 1 is more circle-like). Clamp to 1.0 since
    # perimeter is pixel-approximate and may otherwise produce slight >1 values.
    compactness = 4 * np.pi * area_px / (perimeter**2 + 1e-6)
    compactness = min(compactness, 1.0)

    # Rectangularity: fraction of axis-aligned bounding box occupied by the field.
    rectangularity = area_px / (axis_bbox_area_px + 1e-6)

    # Rotation-invariant aspect ratio from intrinsic principal axes.
    aspect_ratio = minor_axis_length / (major_axis_length + 1e-6)

    # Explicit orientation signal.
    cardinal_alignment = intrinsic_bbox_area_px / (axis_bbox_area_px + 1e-6)

    # Edge density: boundary complexity per unit area.
    edge_density = perimeter / (area_px + 1e-6)

    # Return only numeric primitives for clean downstream DataFrame serialization.
    return {
        "field_id": int(field_id),
        "area": area_sqm,
        "area_px": area_px,
        "perimeter": perimeter,
        "compactness": float(compactness),
        "rectangularity": float(rectangularity),
        "aspect_ratio": float(aspect_ratio),
        "edge_density": float(edge_density),
        "cardinal_alignment": float(cardinal_alignment),
    }