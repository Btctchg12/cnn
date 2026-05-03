from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FTWCNNDataset5000(Dataset):
    def __init__(
        self,
        csv_path,
        data_root=None,
        label_col="cluster",
        use_window_a=True,
        use_window_b=True,
        return_metadata=False,
        normalize=True,
        image_size=(224, 224),
    ):
        # This file is located at:
        # src/models/cnn/experiments_5000/cnn_dataset_5000.py
        # project root is 4 levels above this file
        self.project_root = Path(__file__).resolve().parents[4]

        if data_root is None:
            self.data_root = self.project_root / "parent"
        else:
            self.data_root = Path(data_root)

        self.csv_path = Path(csv_path)

        self.label_col = label_col
        self.use_window_a = use_window_a
        self.use_window_b = use_window_b
        self.return_metadata = return_metadata
        self.normalize = normalize
        self.image_size = image_size

        if not self.use_window_a and not self.use_window_b:
            raise ValueError("At least one of use_window_a or use_window_b must be True.")

        print("\nBuilding 5000 CNN dataset")
        print("-" * 70)
        print("Project root:", self.project_root)
        print("Data root:", self.data_root)
        print("CSV path:", self.csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root folder not found: {self.data_root}")

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
        df[self.label_col] = df[self.label_col].astype(int)

        self.samples = []
        skipped = 0
        skipped_examples = []

        for _, row in df.iterrows():
            chip_id = row["chip_id"]
            country = row["country"]
            label = int(row[self.label_col])

            a_path = None
            b_path = None

            if self.use_window_a:
                a_path = self._find_tif_path(country, chip_id, "window_a")
                if a_path is None:
                    skipped += 1
                    if len(skipped_examples) < 5:
                        skipped_examples.append((country, chip_id, "window_a"))
                    continue

            if self.use_window_b:
                b_path = self._find_tif_path(country, chip_id, "window_b")
                if b_path is None:
                    skipped += 1
                    if len(skipped_examples) < 5:
                        skipped_examples.append((country, chip_id, "window_b"))
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
                "No valid samples found. Check csv_path, parent folder, country names, and tif file names."
            )

        print("CSV rows:", len(df))
        print("Valid samples:", len(self.samples))
        print("Skipped rows:", skipped)

        if skipped_examples:
            print("First skipped examples:")
            for example in skipped_examples:
                print(example)

        labels = self.get_labels()
        print("Label counts:")
        print(pd.Series(labels).value_counts().sort_index())

    def _find_tif_path(self, country, chip_id, window_name):
        country = str(country)
        chip_id = str(chip_id)

        possible_countries = [
            country,
            country.lower(),
            country.upper(),
        ]

        possible_names = [
            f"{chip_id}.tif",
            f"{chip_id}.tiff",
        ]

        for country_name in possible_countries:
            window_dir = self.data_root / country_name / "s2_images" / window_name

            for name in possible_names:
                path = window_dir / name
                if path.exists():
                    return path

        return None

    def __len__(self):
        return len(self.samples)

    def _read_tif(self, tif_path):
        with rasterio.open(tif_path) as src:
            image = src.read().astype(np.float32)

        return image

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_parts = []

        if self.use_window_a:
            image_a = self._read_tif(sample["window_a_path"])
            image_parts.append(image_a)

        if self.use_window_b:
            image_b = self._read_tif(sample["window_b_path"])
            image_parts.append(image_b)

        if len(image_parts) == 1:
            image = image_parts[0]
        else:
            # Combine window_a and window_b along channel dimension
            image = np.concatenate(image_parts, axis=0)

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize:
            image = image / 10000.0

        image = torch.tensor(image, dtype=torch.float32)

        if self.image_size is not None:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        label = torch.tensor(sample["label"], dtype=torch.long)

        if self.return_metadata:
            return image, label, sample["chip_id"], sample["country"]

        return image, label

    def get_labels(self):
        return [sample["label"] for sample in self.samples]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[4]

    train_csv = project_root / "data" / "split" / "train_5000.csv"
    val_csv = project_root / "data" / "split" / "validation_5000.csv"
    test_csv = project_root / "data" / "split" / "test_5000.csv"

    print("\nTesting train dataset")
    train_dataset = FTWCNNDataset5000(
        csv_path=train_csv,
        data_root=project_root / "parent",
        label_col="cluster",
        use_window_a=True,
        use_window_b=True,
        return_metadata=True,
        normalize=True,
        image_size=(224, 224),
    )

    image, label, chip_id, country = train_dataset[0]

    print("\nSample check")
    print("-" * 70)
    print("Dataset length:", len(train_dataset))
    print("Image shape:", image.shape)
    print("Label:", label)
    print("Chip ID:", chip_id)
    print("Country:", country)

    print("\nTesting validation dataset")
    val_dataset = FTWCNNDataset5000(
        csv_path=val_csv,
        data_root=project_root / "parent",
        label_col="cluster",
        use_window_a=True,
        use_window_b=True,
        return_metadata=True,
        normalize=True,
        image_size=(224, 224),
    )

    print("Validation dataset length:", len(val_dataset))

    print("\nTesting test dataset")
    test_dataset = FTWCNNDataset5000(
        csv_path=test_csv,
        data_root=project_root / "parent",
        label_col="cluster",
        use_window_a=True,
        use_window_b=True,
        return_metadata=True,
        normalize=True,
        image_size=(224, 224),
    )

    print("Test dataset length:", len(test_dataset))