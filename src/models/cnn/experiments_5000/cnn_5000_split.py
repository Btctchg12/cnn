from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# This file is located at:
# src/models/cnn/experiments_5000/cnn_5000_split.py
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def print_distribution(name, df):
    print("\n" + "=" * 70)
    print(f"{name} size: {len(df)}")
    print("=" * 70)

    print("\nCluster counts:")
    print(df["cluster"].value_counts().sort_index())

    print("\nCluster distribution (%):")
    print((df["cluster"].value_counts(normalize=True).sort_index() * 100).round(2))

    print("\nCountry counts:")
    print(df["country"].value_counts().sort_index())

    print("\nCountry distribution (%):")
    print((df["country"].value_counts(normalize=True).sort_index() * 100).round(2))


def main():
    original_csv = (
        PROJECT_ROOT
        / "src"
        / "models"
        / "cnn"
        / "labeled_geometry_features.csv"
    )

    if not original_csv.exists():
        raise FileNotFoundError(f"Original CSV not found: {original_csv}")

    df = pd.read_csv(original_csv)

    required_cols = {"chip_id", "country", "cluster"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[["chip_id", "country", "cluster"]].copy()

    df["chip_id"] = df["chip_id"].astype(str)
    df["country"] = df["country"].astype(str)
    df["cluster"] = df["cluster"].astype(int)

    # Make split more stable when using the same random_state
    df = df.sort_values("chip_id").reset_index(drop=True)

    # First split:
    # 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=df["cluster"],
    )

    # Second split:
    # temp 20% -> 10% validation, 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=temp_df["cluster"],
    )

    split_dir = PROJECT_ROOT / "data" / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train_5000.csv"
    val_path = split_dir / "validation_5000.csv"
    test_path = split_dir / "test_5000.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nOriginal CSV:", original_csv)
    print("Saved train split to:", train_path)
    print("Saved validation split to:", val_path)
    print("Saved test split to:", test_path)

    print_distribution("Original full dataset", df)
    print_distribution("Train 5000 set", train_df)
    print_distribution("Validation 5000 set", val_df)
    print_distribution("Test 5000 set", test_df)


if __name__ == "__main__":
    main()