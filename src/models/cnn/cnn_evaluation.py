from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.models.cnn.cnn_dataset import FTWCNNDataset
from src.models.cnn.cnn_model import SimpleCNN



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 固定只用 window_a + window_b
    use_window_a = True
    use_window_b = True
    in_channels = 8

    test_csv = PROJECT_ROOT / "data" / "split" / "test.csv"

    dataset = FTWCNNDataset(
        csv_path=test_csv,
        label_col="cluster",
        use_window_a=use_window_a,
        use_window_b=use_window_b,
        return_metadata=False,
    )

    print("Test samples:", len(dataset))

    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = SimpleCNN(in_channels=in_channels, num_classes=3).to(device)

    model_path = PROJECT_ROOT / "src" / "models" / "cnn" / "simple_cnn_ab.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx % 50 == 0:
                print(f"Test batch {batch_idx}/{len(test_loader)}")

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    print("\n===== TEST RESULT =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    


if __name__ == "__main__":
    main()