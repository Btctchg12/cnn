from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()

# evaluate.py is located at:
# src/models/cnn/experiments_2300/simple_cnn/evaluate.py
PROJECT_ROOT = CURRENT_FILE.parents[5]
EXPERIMENT_DIR = CURRENT_FILE.parents[1]

sys.path.append(str(EXPERIMENT_DIR))

from cnn_dataset_2300 import FTWCNNDataset2300
from model import SimpleCNN


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

SPLIT_TO_EVALUATE = "test"
# options:
# "validation"
# "test"

DATA_ROOT = PROJECT_ROOT / "parent"

VAL_CSV = PROJECT_ROOT / "data" / "split" / "validation_2300.csv"
TEST_CSV = PROJECT_ROOT / "data" / "split" / "test_2300.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "experiments_2300" / "simple_cnn"
CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"

OUTPUT_DIR = MODEL_DIR / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 0

CLASS_NAMES = ["cluster_0", "cluster_1", "cluster_2"]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_eval_csv(split_name):
    if split_name == "validation":
        return VAL_CSV
    elif split_name == "test":
        return TEST_CSV
    else:
        raise ValueError("SPLIT_TO_EVALUATE must be either 'validation' or 'test'.")


def evaluate_model(model, dataloader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_chip_ids = []
    all_countries = []

    total_samples = len(dataloader.dataset)
    processed_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, labels, chip_ids, countries = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            all_chip_ids.extend(chip_ids)
            all_countries.extend(countries)

            processed_samples += images.size(0)
            remaining_samples = total_samples - processed_samples

            print(
                f"Evaluation batch {batch_idx + 1}/{len(dataloader)} "
                f"| processed: {processed_samples}/{total_samples} "
                f"| remaining: {remaining_samples}"
            )

    accuracy = accuracy_score(all_labels, all_preds)

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    per_class_recall = recall_score(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    conf_mat = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(NUM_CLASSES)),
    )

    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        all_labels,
        all_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
        "per_class_recall": per_class_recall.tolist(),
        "confusion_matrix": conf_mat.tolist(),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "labels": all_labels,
        "predictions": all_preds,
        "probabilities": all_probs,
        "chip_ids": all_chip_ids,
        "countries": all_countries,
    }

    return results


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    device = get_device()

    eval_csv = get_eval_csv(SPLIT_TO_EVALUATE)

    print("\n" + "=" * 70)
    print("Evaluating SimpleCNN on 2300 dataset")
    print("=" * 70)
    print("Project root:", PROJECT_ROOT)
    print("Evaluation split:", SPLIT_TO_EVALUATE)
    print("Evaluation CSV:", eval_csv)
    print("Data root:", DATA_ROOT)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Output dir:", OUTPUT_DIR)
    print("Device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Please run train.py first."
        )

    eval_dataset = FTWCNNDataset2300(
        csv_path=eval_csv,
        data_root=DATA_ROOT,
        label_col="cluster",
        use_window_a=True,
        use_window_b=True,
        return_metadata=True,
        normalize=True,
        image_size=IMAGE_SIZE,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    sample_image, sample_label, sample_chip_id, sample_country = eval_dataset[0]
    in_channels = sample_image.shape[0]

    print("\nInput check")
    print("-" * 70)
    print("Sample image shape:", sample_image.shape)
    print("Input channels:", in_channels)
    print("Sample label:", sample_label.item())
    print("Sample chip_id:", sample_chip_id)
    print("Sample country:", sample_country)

    model = SimpleCNN(
        in_channels=in_channels,
        num_classes=NUM_CLASSES,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nLoaded checkpoint")
    print("-" * 70)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Best validation macro-F1:", checkpoint.get("best_val_macro_f1"))

    results = evaluate_model(
        model=model,
        dataloader=eval_loader,
        device=device,
    )

    print("\nEvaluation summary")
    print("-" * 70)
    print(f"Split: {SPLIT_TO_EVALUATE}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print(f"Weighted-F1: {results['weighted_f1']:.4f}")
    print(f"Balanced accuracy: {results['balanced_accuracy']:.4f}")
    print("Per-class recall:", results["per_class_recall"])
    print("Confusion matrix:")
    print(np.array(results["confusion_matrix"]))

    print("\nPer-class precision / recall / F1")
    print("-" * 70)

    report = results["classification_report"]

    for class_name in CLASS_NAMES:
        print(
            f"{class_name} | "
            f"precision: {report[class_name]['precision']:.4f} | "
            f"recall: {report[class_name]['recall']:.4f} | "
            f"f1-score: {report[class_name]['f1-score']:.4f} | "
            f"support: {int(report[class_name]['support'])}"
        )

    print("\nFull classification report")
    print("-" * 70)
    print(results["classification_report_text"])

    probs = np.array(results["probabilities"])

    prediction_df = pd.DataFrame({
        "chip_id": results["chip_ids"],
        "country": results["countries"],
        "true_label": results["labels"],
        "predicted_label": results["predictions"],
        "prob_cluster_0": probs[:, 0],
        "prob_cluster_1": probs[:, 1],
        "prob_cluster_2": probs[:, 2],
    })

    prediction_path = OUTPUT_DIR / f"{SPLIT_TO_EVALUATE}_predictions.csv"
    metrics_path = OUTPUT_DIR / f"{SPLIT_TO_EVALUATE}_metrics.json"
    report_csv_path = OUTPUT_DIR / f"{SPLIT_TO_EVALUATE}_classification_report.csv"

    prediction_df.to_csv(prediction_path, index=False)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(report_csv_path)

    metrics_to_save = {
        "split": SPLIT_TO_EVALUATE,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_macro_f1_from_training": checkpoint.get("best_val_macro_f1"),
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "balanced_accuracy": results["balanced_accuracy"],
        "per_class_recall": results["per_class_recall"],
        "confusion_matrix": results["confusion_matrix"],
        "classification_report": results["classification_report"],
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print("\nSaved evaluation predictions to:", prediction_path)
    print("Saved evaluation metrics to:", metrics_path)
    print("Saved classification report to:", report_csv_path)


if __name__ == "__main__":
    main()