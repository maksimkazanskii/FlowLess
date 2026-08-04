import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.models.mlp import MLP
from src.datasets.split_mnist import SplitMNIST
from geometry_legacy.density import DensityAnalyzer
from geometry_legacy.specturm import SpectralAnalyzer



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3

MEMORY_PER_TASK = 20


class ReplayBuffer:

    def __init__(self):

        self.x = []
        self.y = []

    def add_dataset(self, dataset):

        idx = np.random.choice(
            len(dataset),
            min(MEMORY_PER_TASK, len(dataset)),
            replace=False
        )

        for i in idx:

            x, y = dataset[i]

            self.x.append(x.clone())
            self.y.append(int(y))

    def sample(self, batch_size):

        if len(self.x) == 0:
            return None, None

        idx = np.random.choice(
            len(self.x),
            min(batch_size, len(self.x)),
            replace=False
        )

        x = torch.stack(
            [self.x[i] for i in idx]
        )

        y = torch.tensor(
            [self.y[i] for i in idx],
            dtype=torch.long
        )

        return x, y


def train_task(
        model,
        dataset,
        replay_buffer=None,
        snapshot_callback=None
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):

        running_loss = 0.0

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            loss = criterion(
                model(x),
                y
            )

            if replay_buffer is not None:

                rx, ry = replay_buffer.sample(
                    2 * len(x)
                )

                if rx is not None:

                    rx = rx.to(DEVICE)
                    ry = ry.to(DEVICE)

                    loss += 2.0 * criterion(
                        model(rx),
                        ry
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss: {running_loss/len(loader):.4f}"
        )
        if snapshot_callback is not None:
            snapshot_callback(epoch)


@torch.no_grad()
def evaluate(model, dataset):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += len(y)

    return 100.0 * correct / total


@torch.no_grad()
@torch.no_grad()
def collect_probe_state(model, dataset):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    model.eval()

    features = {
        "layer1": [],
        "layer2": [],
        "layer3": []
    }

    labels = []
    preds = []
    prob_true = []
    margins = []
    correct = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        f = model.features(x)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        pred = logits.argmax(dim=1)

        true_prob = probs[torch.arange(len(y)), y]

        logits_true = logits[torch.arange(len(y)), y]
        logits_tmp = logits.clone()
        logits_tmp[torch.arange(len(y)), y] = -1e9
        best_other = logits_tmp.max(dim=1).values
        margin = logits_true - best_other

        for key in features:
            features[key].append(f[key].cpu().numpy())

        labels.append(y.cpu().numpy())
        preds.append(pred.cpu().numpy())
        prob_true.append(true_prob.cpu().numpy())
        margins.append(margin.cpu().numpy())
        correct.append((pred == y).cpu().numpy())

    for key in features:
        features[key] = np.concatenate(features[key], axis=0)

    return {
        "features": features,
        "labels": np.concatenate(labels, axis=0),
        "preds": np.concatenate(preds, axis=0),
        "prob_true": np.concatenate(prob_true, axis=0),
        "margin": np.concatenate(margins, axis=0),
        "correct": np.concatenate(correct, axis=0).astype(np.int64),
    }



def save_snapshot(
        model,
        probe_datasets,
        snapshot_id,
        density_analyzer,
        spectral_analyzer,
        save_root
):

    snapshot_dir = (
            save_root
            / f"snapshot_{snapshot_id}"
    )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for probe_task_id, probe_dataset in enumerate(
            probe_datasets
    ):

        probe_dir = (
                snapshot_dir
                / f"probe_task_{probe_task_id}"
        )

        probe_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        state = collect_probe_state(
            model,
            probe_dataset
        )

        features = state["features"]

        np.save(probe_dir / "labels.npy", state["labels"])
        np.save(probe_dir / "preds.npy", state["preds"])
        np.save(probe_dir / "prob_true.npy", state["prob_true"])
        np.save(probe_dir / "margin.npy", state["margin"])
        np.save(probe_dir / "correct.npy", state["correct"])

        stats = {}

        for layer_name, Z in features.items():

            np.save(
                probe_dir
                / f"{layer_name}.npy",
                Z
            )

            density_stats = (
                density_analyzer.compute(Z)
            )

            spectral_stats = (
                spectral_analyzer.compute(Z)
            )

            stats[layer_name] = {
                "density_mean":
                    float(
                        density_stats["mean"]
                    ),
                "density_std":
                    float(
                        density_stats["std"]
                    ),
                "effective_dim":
                    float(
                        spectral_stats[
                            "effective_dim"
                        ]
                    )
            }

        with open(
                probe_dir / "stats.json",
                "w"
        ) as f:

            json.dump(
                stats,
                f,
                indent=4
            )

    print(
        f"Saved snapshot "
        f"{snapshot_id}"
    )

def main():
    dataset = SplitMNIST()
    model = MLP().to(DEVICE)

    replay_buffer = ReplayBuffer()
    density_analyzer = DensityAnalyzer()
    spectral_analyzer = SpectralAnalyzer()

    save_root = Path(
        "data/mnist/mlp/replay"
    )
    save_root.mkdir(parents=True, exist_ok=True)

    num_tasks = dataset.num_tasks()

    acc_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

    probe_datasets = []

    for task_id in range(num_tasks):
        probe_taskset, _ = dataset.get_task(task_id)
        probe_datasets.append(probe_taskset)

    save_snapshot(
        model=model,
        probe_datasets=[probe_datasets[0]],
        snapshot_id="init",
        density_analyzer=density_analyzer,
        spectral_analyzer=spectral_analyzer,
        save_root=save_root
    )
    for current_task in range(num_tasks):
        train_taskset, _ = dataset.get_task(current_task)
        print("Task", current_task)
        print("Classes:", dataset.get_classes(current_task))

        x, y = next(iter(DataLoader(train_taskset, batch_size=20)))
        print("Unique labels:", torch.unique(y))
        print()
        print("=" * 60)
        print(f"TRAINING TASK {current_task + 1}")
        print(f"CLASSES: {dataset.get_classes(current_task)}")
        print("=" * 60)


        train_task(
            model=model,
            dataset=train_taskset,
            replay_buffer=replay_buffer,
            snapshot_callback=lambda epoch: save_snapshot(
                model=model,
                probe_datasets=probe_datasets[:current_task + 1],
                snapshot_id=f"task{current_task}_epoch{epoch}",
                density_analyzer=density_analyzer,
                spectral_analyzer=spectral_analyzer,
                save_root=save_root
            )
        )
        replay_buffer.add_dataset(
            train_taskset
        )

        print(
            "Replay size:",
            len(replay_buffer.x)
        )

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