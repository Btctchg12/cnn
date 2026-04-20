from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    csv_path = PROJECT_ROOT / "data" / "labeled_geometry_features.csv"
    df = pd.read_csv(csv_path)

    # 只保留需要的列
    df = df[["chip_id", "country", "cluster"]].copy()

    # 90% train, 10% test
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    split_dir = PROJECT_ROOT / "data" / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))
    print("Saved train split to:", train_path)
    print("Saved test split to:", test_path)


if __name__ == "__main__":
    main()