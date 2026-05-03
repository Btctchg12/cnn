from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def proportional_downsample(df, target_size=2300, random_state=42):
    """
    Downsample the full dataset to around target_size,
    while keeping country + cluster proportions similar to the original dataset.
    """
    group_cols = ["country", "cluster"]

    group_counts = df.groupby(group_cols).size().reset_index(name="count")
    total_size = len(df)

    group_counts["exact_n"] = group_counts["count"] / total_size * target_size
    group_counts["sample_n"] = np.floor(group_counts["exact_n"]).astype(int)
    remaining = target_size - group_counts["sample_n"].sum()
    group_counts["remainder"] = group_counts["exact_n"] - group_counts["sample_n"]
    group_counts = group_counts.sort_values("remainder", ascending=False).reset_index(drop=True)

    if remaining > 0:
        group_counts.loc[:remaining - 1, "sample_n"] += 1

    sampled_parts = []

    for _, row in group_counts.iterrows():
        country = row["country"]
        cluster = row["cluster"]
        n = int(row["sample_n"])

        group_df = df[
            (df["country"] == country) &
            (df["cluster"] == cluster)
        ]

        if n > 0:
            sampled_parts.append(
                group_df.sample(n=n, random_state=random_state)
            )

    sampled_df = pd.concat(sampled_parts)

    sampled_df = sampled_df.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    return sampled_df


def print_distribution(name, df):
    print(f"\n{name} size:", len(df))

    print(f"\n{name} cluster distribution:")
    print((df["cluster"].value_counts(normalize=True).sort_index() * 100).round(2))

    print(f"\n{name} country distribution:")
    print((df["country"].value_counts(normalize=True).sort_index() * 100).round(2))


def main():
    current_dir = Path(__file__).resolve().parent
    csv_path = current_dir / "labeled_geometry_features.csv"

    df = pd.read_csv(csv_path)
    df = df[["chip_id", "country", "cluster"]].copy()
    df = df.sort_values("chip_id").reset_index(drop=True)

    # Step 1: create proportional 2300-row dataset
    sampled_df = proportional_downsample(
        df,
        target_size=2300,
        random_state=42
    )

    subset_path = current_dir / "labeled_geometry_features_2300.csv"
    sampled_df.to_csv(subset_path, index=False)

    train_df, temp_df = train_test_split(
        sampled_df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=sampled_df["cluster"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=temp_df["cluster"]
    )

    split_dir = PROJECT_ROOT / "data" / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train_2300.csv"
    val_path = split_dir / "validation_2300.csv"
    test_path = split_dir / "test_2300.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Original size:", len(df))
    print("Sampled size:", len(sampled_df))

    print_distribution("Original dataset", df)
    print_distribution("Sampled 2300 dataset", sampled_df)
    print_distribution("Train set", train_df)
    print_distribution("Validation set", val_df)
    print_distribution("Test set", test_df)

    print("\nSaved 2300 subset to:", subset_path)
    print("Saved train split to:", train_path)
    print("Saved validation split to:", val_path)
    print("Saved test split to:", test_path)


if __name__ == "__main__":
    main()