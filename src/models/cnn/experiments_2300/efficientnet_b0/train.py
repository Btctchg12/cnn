from pathlib import Path
import sys
import json
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
)

# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()

# train.py is located at:
# src/models/cnn/experiments_2300/efficientnet_b0/train.py
PROJECT_ROOT = CURRENT_FILE.parents[5]
EXPERIMENT_DIR = CURRENT_FILE.parents[1]

sys.path.append(str(EXPERIMENT_DIR))

from cnn_dataset_2300 import FTWCNNDataset2300
from model import EfficientNetB0Classifier


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

TRAIN_CSV = PROJECT_ROOT / "data" / "split" / "train_2300.csv"
VAL_CSV = PROJECT_ROOT / "data" / "split" / "validation_2300.csv"
DATA_ROOT = PROJECT_ROOT / "parent"

OUTPUT_DIR = PROJECT_ROOT / "models" / "experiments_2300" / "efficientnet_b0"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 30

# EfficientNet-B0 is pretrained, so use a smaller learning rate.
LEARNING_RATE = 5e-5

WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
USE_WEIGHTED_LOSS = True
PRETRAINED = True

IMAGE_SIZE = (224, 224)
NUM_WORKERS = 0


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def compute_class_weights(dataset, num_classes):
    labels = np.array(dataset.get_labels())
    counts = np.bincount(labels, minlength=num_classes)

    total = len(labels)
    weights = []

    for class_id in range(num_classes):
        count = counts[class_id]

        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))

    return torch.tensor(weights, dtype=torch.float32), counts


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()

    total_samples = len(dataloader.dataset)
    processed_samples = 0

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for batch_idx, batch in enumerate(dataloader):
        images, labels, chip_ids, countries = batch

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item() * images.size(0)

        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

        processed_samples += images.size(0)
        remaining_samples = total_samples - processed_samples

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(
                f"Epoch {epoch} | Train batch {batch_idx + 1}/{len(dataloader)} "
                f"| processed: {processed_samples}/{total_samples} "
                f"| remaining: {remaining_samples}"
            )

    epoch_loss = running_loss / total_samples
    epoch_accuracy = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_accuracy


def evaluate(model, dataloader, criterion, device, epoch, split_name="Validation"):
    model.eval()

    total_samples = len(dataloader.dataset)
    processed_samples = 0

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_chip_ids = []
    all_countries = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, labels, chip_ids, countries = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)

            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_chip_ids.extend(chip_ids)
            all_countries.extend(countries)

            processed_samples += images.size(0)
            remaining_samples = total_samples - processed_samples

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(
                    f"Epoch {epoch} | {split_name} batch {batch_idx + 1}/{len(dataloader)} "
                    f"| tested: {processed_samples}/{total_samples} "
                    f"| remaining: {remaining_samples}"
                )

    epoch_loss = running_loss / total_samples

    accuracy = accuracy_score(all_labels, all_preds)

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
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

    results = {
        "loss": epoch_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "per_class_recall": per_class_recall.tolist(),
        "confusion_matrix": conf_mat.tolist(),
        "chip_ids": all_chip_ids,
        "countries": all_countries,
        "labels": all_labels,
        "predictions": all_preds,
    }

    return results


# ------------------------------------------------------------
# Main training
# ------------------------------------------------------------

def main():
    set_seed(RANDOM_SEED)

    device = get_device()

    print("\n" + "=" * 70)
    print("Training EfficientNet-B0 on 2300 dataset")
    print("=" * 70)
    print("Project root:", PROJECT_ROOT)
    print("Train CSV:", TRAIN_CSV)
    print("Validation CSV:", VAL_CSV)
    print("Data root:", DATA_ROOT)
    print("Output dir:", OUTPUT_DIR)
    print("Device:", device)
    print("Pretrained:", PRETRAINED)

    train_dataset = FTWCNNDataset2300(
        csv_path=TRAIN_CSV,
        data_root=DATA_ROOT,
        label_col="cluster",
        use_window_a=True,
        use_window_b=True,
        return_metadata=True,
        normalize=True,
        image_size=IMAGE_SIZE,
    )

    val_dataset = FTWCNNDataset2300(
        csv_path=VAL_CSV,
        data_root=DATA_ROOT,
        label_col="cluster",
        use_window_a=True,
        use_window_b=True,
        return_metadata=True,
        normalize=True,
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    sample_image, sample_label, sample_chip_id, sample_country = train_dataset[0]
    in_channels = sample_image.shape[0]

    print("\nInput check")
    print("-" * 70)
    print("Sample image shape:", sample_image.shape)
    print("Input channels:", in_channels)
    print("Sample label:", sample_label.item())
    print("Sample chip_id:", sample_chip_id)
    print("Sample country:", sample_country)

    model = EfficientNetB0Classifier(
        in_channels=in_channels,
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
    ).to(device)

    if USE_WEIGHTED_LOSS:
        class_weights, class_counts = compute_class_weights(train_dataset, NUM_CLASSES)
        class_weights = class_weights.to(device)

        print("\nUsing weighted CrossEntropyLoss")
        print("Train class counts:", class_counts)
        print("Class weights:", class_weights.detach().cpu().numpy())

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("\nUsing unweighted CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config = {
        "model": "EfficientNetB0Classifier",
        "dataset": "2300 proportional subset",
        "train_csv": str(TRAIN_CSV),
        "validation_csv": str(VAL_CSV),
        "data_root": str(DATA_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "num_classes": NUM_CLASSES,
        "in_channels": in_channels,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "random_seed": RANDOM_SEED,
        "weighted_loss": USE_WEIGHTED_LOSS,
        "pretrained": PRETRAINED,
        "image_size": IMAGE_SIZE,
        "device": str(device),
    }

    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    history = []
    best_val_macro_f1 = -1.0
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 70)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 70)

        start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_results = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            split_name="Validation",
        )

        epoch_time = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_results["loss"],
            "val_accuracy": val_results["accuracy"],
            "val_macro_f1": val_results["macro_f1"],
            "val_balanced_accuracy": val_results["balanced_accuracy"],
            "val_recall_cluster_0": val_results["per_class_recall"][0],
            "val_recall_cluster_1": val_results["per_class_recall"][1],
            "val_recall_cluster_2": val_results["per_class_recall"][2],
            "epoch_time_seconds": epoch_time,
        }

        history.append(row)

        history_df = pd.DataFrame(history)
        history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

        print("\nEpoch summary")
        print("-" * 70)
        print(f"Epoch: {epoch}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Validation loss: {val_results['loss']:.4f}")
        print(f"Validation accuracy: {val_results['accuracy']:.4f}")
        print(f"Validation macro-F1: {val_results['macro_f1']:.4f}")
        print(f"Validation balanced accuracy: {val_results['balanced_accuracy']:.4f}")
        print("Validation per-class recall:", val_results["per_class_recall"])
        print("Validation confusion matrix:")
        print(np.array(val_results["confusion_matrix"]))
        print(f"Epoch time: {epoch_time:.2f} seconds")

        if val_results["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_results["macro_f1"]
            best_epoch = epoch

            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "config": config,
            }

            torch.save(best_checkpoint, OUTPUT_DIR / "best_model.pth")

            best_val_pred_df = pd.DataFrame({
                "chip_id": val_results["chip_ids"],
                "country": val_results["countries"],
                "true_label": val_results["labels"],
                "predicted_label": val_results["predictions"],
            })

            best_val_pred_df.to_csv(
                OUTPUT_DIR / "best_validation_predictions.csv",
                index=False,
            )

            print(
                f"\nNew best model saved at epoch {epoch}. "
                f"Best validation macro-F1: {best_val_macro_f1:.4f}"
            )

        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "config": config,
        }

        torch.save(last_checkpoint, OUTPUT_DIR / "last_model.pth")

    print("\n" + "=" * 70)
    print("Training finished")
    print("=" * 70)
    print("Best epoch:", best_epoch)
    print("Best validation macro-F1:", best_val_macro_f1)
    print("Saved best model to:", OUTPUT_DIR / "best_model.pth")
    print("Saved last model to:", OUTPUT_DIR / "last_model.pth")
    print("Saved training history to:", OUTPUT_DIR / "training_history.csv")
    print("Saved best validation predictions to:", OUTPUT_DIR / "best_validation_predictions.csv")


if __name__ == "__main__":
    main()