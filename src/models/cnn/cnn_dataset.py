from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


class FTWCNNDataset(Dataset):
    def __init__(
        self,
        csv_path,
        countries=None,
        label_col="cluster",
        use_window_a=True,
        use_window_b=True,
        return_metadata=False,
        transform=None,
        normalize=True,
    ):
        self.project_root = Path(__file__).resolve().parents[3]
        self.data_root = self.project_root.parent / "parent"
        self.csv_path = Path(csv_path)
        self.label_col = label_col
        self.use_window_a = use_window_a
        self.use_window_b = use_window_b
        self.return_metadata = return_metadata
        self.transform = transform
        self.normalize = normalize

        if not self.use_window_a and not self.use_window_b:
            raise ValueError("At least one of use_window_a or use_window_b must be True.")

        if not self.csv_path.is_absolute():
            self.csv_path = (self.project_root / self.csv_path).resolve()

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        required_cols = {"chip_id", "country", self.label_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        df["chip_id"] = df["chip_id"].astype(str)
        df["country"] = df["country"].astype(str)

        if countries is not None:
            countries = set(countries)
            df = df[df["country"].isin(countries)].copy()

        self.samples = []

        for _, row in df.iterrows():
            chip_id = row["chip_id"]
            country = row["country"]
            label = int(row[self.label_col])

            a_path = None
            b_path = None

            if self.use_window_a:
                a_path = self.data_root / country / "s2_images" / "window_a" / f"{chip_id}.tif"
                if not a_path.exists():
                    continue

            if self.use_window_b:
                b_path = self.data_root / country / "s2_images" / "window_b" / f"{chip_id}.tif"
                if not b_path.exists():
                    continue

            self.samples.append(
                {
                    "chip_id": chip_id,
                    "country": country,
                    "label": label,
                    "window_a_path": a_path,
                    "window_b_path": b_path,
                }
            )

        if len(self.samples) == 0:
            raise ValueError(
                "No valid samples found. Check csv_path, label_col, parent folder, and file structure."
            )

        print(f"Built dataset with {len(self.samples)} samples.")
        print(f"Data root: {self.data_root}")
        print(f"CSV path: {self.csv_path}")

    def __len__(self):
        return len(self.samples)

    def _read_tif(self, tif_path: Path) -> np.ndarray:
        with rasterio.open(tif_path) as src:
            img = src.read().astype(np.float32)   # (C, H, W)
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_parts = []

        if self.use_window_a:
            img_a = self._read_tif(sample["window_a_path"])
            image_parts.append(img_a)

        if self.use_window_b:
            img_b = self._read_tif(sample["window_b_path"])
            image_parts.append(img_b)

        if len(image_parts) == 1:
            image = image_parts[0]
        else:
            image = np.concatenate(image_parts, axis=0)

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize:
            image = image / 10000.0

        if self.transform is not None:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)

        if self.return_metadata:
            return image, label, sample["chip_id"], sample["country"]

        return image, label


if __name__ == "__main__":
    dataset = FTWCNNDataset(
        csv_path="data/labeled_geometry_features.csv",
        label_col="cluster",
        use_window_a=True,
        use_window_b=True,
        return_metadata=True,
    )

    print("Dataset length:", len(dataset))

    image, label, chip_id, country = dataset[0]
    print("Image shape:", image.shape)
    print("Label:", label)
    print("Chip ID:", chip_id)
    print("Country:", country)