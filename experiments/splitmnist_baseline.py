import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.models.mlp import MLP
from src.datasets.split_mnist import SplitMNIST

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3

# ============================================================
# Train
# ============================================================

def train_task(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}")


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def evaluate(model, dataset):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += len(y)

    return 100.0 * correct / total


# ============================================================
# Main
# ============================================================

def main():
    dataset = SplitMNIST()
    model = MLP().to(DEVICE)

    num_tasks = dataset.num_tasks()
    acc_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

    for current_task in range(num_tasks):
        train_taskset, _ = dataset.get_task(current_task)

        print()
        print("=" * 60)
        print(f"TRAINING TASK {current_task + 1}")
        print(f"CLASSES: {dataset.get_classes(current_task)}")
        print("=" * 60)

        train_task(model, train_taskset)

        print()
        print("Evaluation")

        for eval_task in range(current_task + 1):
            _, test_taskset = dataset.get_task(eval_task)

            acc = evaluate(model, test_taskset)

            acc_matrix[current_task, eval_task] = acc

            print(f"Task {eval_task + 1}: {acc:.2f}%")

    print()
    print("=" * 70)
    print("FORGETTING MATRIX")
    print("=" * 70)

    for row in acc_matrix:
        print(" ".join(f"{v:7.2f}" for v in row))

    print()

    final_acc = acc_matrix[-1].mean()

    print(f"Final Average Accuracy: {final_acc:.2f}%")


if __name__ == "__main__":
    main()