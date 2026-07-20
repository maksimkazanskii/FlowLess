import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.mlp import MLP
from src.datasets.datasets import Dataset

from sklearn.neighbors import NearestNeighbors


def compute_density(Z, k=10):

    if len(Z) <= k:
        return np.ones(len(Z))

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(Z)

    distances, _ = nn.kneighbors(Z)

    distances = distances[:, 1:]

    rho = k / (distances.sum(axis=1) + 1e-12)

    return rho

OUT = Path("data/results/flowless/layers")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3

REPLAY_WEIGHT = 2.0
LAMBDA_FLUX = 0.1
MEMORY_SIZE = 40

LAYER_CONFIGS = [
    ["layer1"],
    ["layer2"],
    ["layer3"],
    ["layer2", "layer3"],
    ["layer1", "layer2", "layer3"],
]

LAMBDA_GRID = [
    0.1,
    0.3,
    1.0,
    3.0,
]




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FluxReplayBuffer:

    def __init__(
            self,
            memory_per_task,
            layers,
            density_memory=False,
    ):
        self.memory_per_task = memory_per_task
        self.layers = layers
        self.density_memory = density_memory


        self.x = []
        self.y = []
        self.z_ref = []
    def __len__(self):
        return len(self.x)

    @torch.no_grad()
    def add_dataset(self, model, dataset):

        if self.memory_per_task <= 0:
            return

        model.eval()

        k = min(self.memory_per_task, len(dataset))

        #
        # Density-aware selection
        #
        if self.density_memory:
            raise NotImplementedError(
            "Density memory not supported in layer ablation"
            )
        #
        # Standard random replay
        #
        else:

            idx = np.random.choice(
                len(dataset),
                size=k,
                replace=False,
            )

        #
        # Store selected samples
        #
        for i in idx:

            x, y = dataset[i]

            xb = x.unsqueeze(0).to(DEVICE)

            features = model.features(xb)

            z = {
                layer:
                    features[layer]
                    .squeeze(0)
                    .detach()
                    .cpu()
                for layer in self.layers
            }

            self.x.append(x.clone())
            self.y.append(int(y))
            self.z_ref.append(z)

    def sample(self, batch_size):
        if len(self.x) == 0:
            return None, None, None

        idx = np.random.choice(
            len(self.x),
            min(batch_size, len(self.x)),
            replace=False,
        )

        x = torch.stack([self.x[i] for i in idx])
        y = torch.tensor([self.y[i] for i in idx], dtype=torch.long)
        z_ref = {}

        for layer in self.layers:
            z_ref[layer] = torch.stack(
                [self.z_ref[i][layer] for i in idx]
            )

        return x, y, z_ref


def flux_loss(model, x, z_ref, layers):

    feats = model.features(x)

    total = 0.0

    for layer in layers:

        z_now = feats[layer]
        z_old = z_ref[layer]

        z_now = F.normalize(z_now, dim=1)
        z_old = F.normalize(z_old, dim=1)

        total += (
                (z_now - z_old) ** 2
        ).sum(dim=1).mean()

    return total / len(layers)

def train_task(
        model,
        dataset,
        replay_buffer,
        use_flux_reg,
        lambda_flux,
        layers,
):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    logs = []

    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_task = 0.0
        running_replay = 0.0
        running_flux = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss_task = criterion(logits, y)
            loss = loss_task

            loss_replay = torch.tensor(0.0, device=DEVICE)
            loss_flux = torch.tensor(0.0, device=DEVICE)

            rx, ry, rz = replay_buffer.sample(2 * len(x))

            if rx is not None:
                rx = rx.to(DEVICE)
                ry = ry.to(DEVICE)
                rz = {
                    layer: tensor.to(DEVICE)
                    for layer, tensor in rz.items()
                }

                loss_replay = criterion(model(rx), ry)
                loss = loss + REPLAY_WEIGHT * loss_replay

                if use_flux_reg:
                    loss_flux = flux_loss(
                        model,
                        rx,
                        rz,
                        layers,
                    )
                    loss = loss + lambda_flux * loss_flux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_task += float(loss_task.item())
            running_replay += float(loss_replay.item())
            running_flux += float(loss_flux.item())
            n_batches += 1

        logs.append({
            "epoch": epoch,
            "loss": running_loss / max(n_batches, 1),
            "task_loss": running_task / max(n_batches, 1),
            "replay_loss": running_replay / max(n_batches, 1),
            "flux_loss": running_flux / max(n_batches, 1),
        })

    return logs


@torch.no_grad()
def evaluate(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
    )

    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x).argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += len(y)

    return 100.0 * correct / max(total, 1)


def forgetting_score(acc_matrix):
    num_tasks = acc_matrix.shape[0]
    vals = []

    for task in range(num_tasks - 1):
        history = acc_matrix[task:, task]
        vals.append(float(np.max(history) - history[-1]))

    return float(np.mean(vals)) if vals else 0.0


def run_condition(
        dataset_name,
        memory_per_task,
        use_flux_reg,
        lambda_flux,
        layers,
        seed,
        out_dir,
        density_memory,
):
    set_seed(seed)
    dataset = Dataset(dataset_name)
    model = MLP().to(DEVICE)
    replay_buffer = FluxReplayBuffer(
        memory_per_task,
        layers=layers,
        density_memory=density_memory,
    )

    num_tasks = dataset.num_tasks()
    acc_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

    train_logs = []

    for current_task in range(num_tasks):
        train_taskset, _ = dataset.get_task(current_task)

        print()
        print("=" * 70)
        print(
            f"seed={seed} memory={memory_per_task} "
            f"flux_reg={int(use_flux_reg)} task={current_task}"
        )
        print("=" * 70)

        logs = train_task(
            model=model,
            dataset=train_taskset,
            replay_buffer=replay_buffer,
            use_flux_reg=use_flux_reg,
            lambda_flux=lambda_flux,
            layers=layers,
        )

        for row in logs:
            row.update({
                "seed": seed,
                "memory_per_task": memory_per_task,
                "use_flux_reg": int(use_flux_reg),
                "task": current_task,
                "lambda_flux": lambda_flux,
            })

        train_logs.extend(logs)

        replay_buffer.add_dataset(model, train_taskset)

        for eval_task in range(current_task + 1):
            _, test_taskset = dataset.get_task(eval_task)
            acc = evaluate(model, test_taskset)
            acc_matrix[current_task, eval_task] = acc
            print(f"eval task {eval_task}: {acc:.2f}")

    final_avg_acc = float(acc_matrix[-1].mean())
    mean_forgetting = forgetting_score(acc_matrix)

    layer_tag = "-".join(layers)

    tag = (
        f"seed{seed}_mem{memory_per_task}_"
        f"layers{layer_tag}_"
        f"flux{int(use_flux_reg)}_lambda{lambda_flux:g}"
    )

    np.save(out_dir / f"{tag}_acc_matrix.npy", acc_matrix)

    with open(out_dir / f"{tag}_train_logs.json", "w") as f:
        json.dump(train_logs, f, indent=4)

    return {
        "dataset": dataset_name,
        "seed": seed,
        "memory_per_task": memory_per_task,
        "layers": "+".join(layers),
        "use_flux_reg": int(use_flux_reg),
        "lambda_flux": lambda_flux,
        "final_avg_acc": final_avg_acc,
        "mean_forgetting": mean_forgetting,
        "replay_size": len(replay_buffer),
    }


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--density_memory",
        action="store_true",
        help="Use density-aware memory selection",
    )
    parser.add_argument(
        "--memory_sizes",
        type=str,
        default="10,20,40,80,160",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
    )

    parser.add_argument(
        "--lambda_flux",
        type=float,
        default=LAMBDA_FLUX,
    )

    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT),
    )

    args = parser.parse_args()

    memory_sizes = parse_int_list(args.memory_sizes)
    seeds = parse_int_list(args.seeds)
    memory_tag = (
        "density"
        if args.density_memory
        else "random"
    )

    out_dir = (
            Path(args.out)
            / args.dataset
            / memory_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []



    for seed in seeds:
        for layers in LAYER_CONFIGS:
            for lambda_flux in LAMBDA_GRID:

                result = run_condition(
                    dataset_name=args.dataset,
                    memory_per_task=MEMORY_SIZE,
                    use_flux_reg=(lambda_flux > 0),
                    lambda_flux=lambda_flux,
                    layers=layers,
                    seed=seed,
                    out_dir=out_dir,
                    density_memory=args.density_memory,
                )

                results.append(result)

                results_csv = out_dir / "results.csv"

                with open(results_csv, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=list(results[0].keys()),
                    )
                    writer.writeheader()
                    writer.writerows(results)

                print(f"saved: {results_csv}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for row in results:
        print(row)

    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
