from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn.cnn_dataset import FTWCNNDataset
from src.models.cnn.cnn_model import SimpleCNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 固定只用 window_a + window_b
    use_window_a = True
    use_window_b = True
    in_channels = 8

    train_csv = PROJECT_ROOT / "data" / "split" / "train.csv"

    dataset = FTWCNNDataset(
        csv_path=train_csv,
        label_col="cluster",
        use_window_a=use_window_a,
        use_window_b=use_window_b,
        return_metadata=False,
    )

    print("Train samples:", len(dataset))

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SimpleCNN(in_channels=in_channels, num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print(f"Training batch {batch_idx}/{len(train_loader)}")

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )

    save_path = PROJECT_ROOT / "src" / "models" / "cnn" / "simple_cnn_ab.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print("\nTraining finished.")
    print("Model saved to:", save_path)


if __name__ == "__main__":
    main()