from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

MODEL_NAMES = [
    "simple_cnn",
    "cnn_bn_dropout",
    "resnet18",
    "efficientnet_b0",
    "resnet50",
]

CLASS_NAMES = ["cluster_0", "cluster_1", "cluster_2"]
LABELS = [0, 1, 2]

summary_rows = []


for model_name in MODEL_NAMES:
    model_dir = (
        PROJECT_ROOT
        / "models"
        / "experiments_2300"
        / model_name
    )

    history_path = model_dir / "training_history.csv"
    pred_path = model_dir / "best_validation_predictions.csv"

    print("\n" + "=" * 70)
    print(f"Model: {model_name}")
    print("=" * 70)

    if not history_path.exists():
        print("training_history.csv not found:", history_path)
        continue

    if not pred_path.exists():
        print("best_validation_predictions.csv not found:", pred_path)
        continue

    history_df = pd.read_csv(history_path)
    best_row = history_df.loc[history_df["val_macro_f1"].idxmax()]

    print("Best validation epoch:", int(best_row["epoch"]))
    print("Validation accuracy from training:", round(best_row["val_accuracy"], 4))
    print("Validation macro-F1 from training:", round(best_row["val_macro_f1"], 4))
    print("Validation balanced accuracy from training:", round(best_row["val_balanced_accuracy"], 4))

    pred_df = pd.read_csv(pred_path)

    y_true = pred_df["true_label"]
    y_pred = pred_df["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=LABELS,
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=LABELS,
        zero_division=0,
    )

    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print("\nRecomputed best validation metrics")
    print("-" * 70)
    print("Accuracy:", round(accuracy, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("Weighted-F1:", round(weighted_f1, 4))
    print("Balanced accuracy:", round(balanced_acc, 4))

    conf_mat = confusion_matrix(y_true, y_pred, labels=LABELS)

    print("\nConfusion matrix:")
    print(conf_mat)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    print("\nPer-class precision / recall / F1")
    print("-" * 70)

    for class_name in CLASS_NAMES:
        print(
            f"{class_name} | "
            f"precision: {report[class_name]['precision']:.4f} | "
            f"recall: {report[class_name]['recall']:.4f} | "
            f"f1-score: {report[class_name]['f1-score']:.4f} | "
            f"support: {int(report[class_name]['support'])}"
        )

    summary_row = {
        "model": model_name,
        "best_validation_epoch": int(best_row["epoch"]),

        "val_accuracy": accuracy,
        "val_macro_f1": macro_f1,
        "val_weighted_f1": weighted_f1,
        "val_balanced_accuracy": balanced_acc,

        "cluster_0_precision": report["cluster_0"]["precision"],
        "cluster_0_recall": report["cluster_0"]["recall"],
        "cluster_0_f1": report["cluster_0"]["f1-score"],
        "cluster_0_support": report["cluster_0"]["support"],

        "cluster_1_precision": report["cluster_1"]["precision"],
        "cluster_1_recall": report["cluster_1"]["recall"],
        "cluster_1_f1": report["cluster_1"]["f1-score"],
        "cluster_1_support": report["cluster_1"]["support"],

        "cluster_2_precision": report["cluster_2"]["precision"],
        "cluster_2_recall": report["cluster_2"]["recall"],
        "cluster_2_f1": report["cluster_2"]["f1-score"],
        "cluster_2_support": report["cluster_2"]["support"],
    }

    summary_rows.append(summary_row)


if len(summary_rows) > 0:
    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        by="val_macro_f1",
        ascending=False
    ).reset_index(drop=True)

    summary_path = (
        PROJECT_ROOT
        / "models"
        / "experiments_2300"
        / "best_validation_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("Overall validation ranking by macro-F1")
    print("=" * 70)

    print(
        summary_df[
            [
                "model",
                "best_validation_epoch",
                "val_accuracy",
                "val_macro_f1",
                "val_weighted_f1",
                "val_balanced_accuracy",
                "cluster_0_f1",
                "cluster_1_f1",
                "cluster_2_f1",
            ]
        ].round(4)
    )

    print("\nSaved summary to:", summary_path)

else:
    print("\nNo model summaries were created. Check whether training files were generated.")