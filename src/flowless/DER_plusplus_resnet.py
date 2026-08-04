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
from src.models.resnet import ResNet18

ALGORITHM = "DER++"
OUT = Path("data/results/flowless_derpp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def get_experiment_config(dataset_name):

    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        return {
            "model_factory": lambda: MLP(),
            "layer": "layer3",
            "epochs": 5,
            "batch_size": 256,
            "eval_batch_size": 512,
            "lr": 1e-3,
            "optimizer": "adam",
            "der_replay_weight": 2.0,
        }

    if dataset_name in {
        "fashion_mnist",
        "fashionmnist",
    }:
        return {
            "model_factory": lambda: MLP(),
            "layer": "layer3",
            "epochs": 5,
            "batch_size": 256,
            "eval_batch_size": 512,
            "lr": 1e-3,
            "optimizer": "adam",
            "der_replay_weight": 2.0,
        }

    if dataset_name == "cifar10":
        return {
            "model_factory": lambda: ResNet18(num_classes=10),
            "layer": "layer4",
            "epochs": 40,
            "batch_size": 128,
            "eval_batch_size": 256,
            "lr": 0.1,
            "optimizer": "sgd",
            "der_replay_weight": 2.0,
        }

    raise ValueError(f"Unsupported dataset: {dataset_name}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FluxReplayBuffer:

    def __init__(self, memory_per_task, layer):
        self.memory_per_task = memory_per_task
        self.layer = layer

        self.x = []
        self.y = []
        self.z_ref = []
        self.logits = []
    def __len__(self):
        return len(self.x)

    @torch.no_grad()
    def add_dataset(self, model, dataset):

        if self.memory_per_task <= 0:
            return

        model.eval()

        k = min(self.memory_per_task, len(dataset))

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

            #
            # Store reference representation
            #
            z = (
                model.features(xb)[self.layer]
                .squeeze(0)
                .detach()
                .cpu()
            )

            #
            # Store teacher logits (DER++)
            #
            logits = (
                model(xb)
                .squeeze(0)
                .detach()
                .cpu()
            )

            self.x.append(x.clone())
            self.y.append(int(y))
            self.z_ref.append(z.clone())
            self.logits.append(logits.clone())

    def sample(self, batch_size):
        if len(self.x) == 0:
            return None

        idx = np.random.choice(
            len(self.x),
            min(batch_size, len(self.x)),
            replace=False,
        )

        x = torch.stack([self.x[i] for i in idx])
        y = torch.tensor([self.y[i] for i in idx], dtype=torch.long)
        z_ref = torch.stack([self.z_ref[i] for i in idx])
        logits = torch.stack([self.logits[i] for i in idx])

        return x, y, z_ref, logits


def flux_loss(model, x, z_ref, layer, normalize=True):
    z_now = model.features(x)[layer]

    if normalize:
        z_now = F.normalize(z_now, dim=1)
        z_ref = F.normalize(z_ref, dim=1)

    return ((z_now - z_ref) ** 2).sum(dim=1).mean()


def train_task(
        model,
        dataset,
        replay_buffer,
        lambda_flux,
        der_replay_weight,
        batch_size,
        epochs,
        lr,
        optimizer_name,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    if optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

    elif optimizer_name == "sgd":

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    scheduler = None

    if optimizer_name == "sgd":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )

    criterion = nn.CrossEntropyLoss()

    model.train()

    logs = []

    for epoch in range(epochs):

        running_loss = 0.0
        running_task = 0.0
        running_mse = 0.0
        running_replay_ce = 0.0
        running_flux = 0.0
        n_batches = 0

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            #
            # Current task loss
            #
            logits = model(x)
            loss_task = criterion(logits, y)

            loss = loss_task

            #
            # Initialize logging variables
            #
            loss_mse = torch.tensor(0.0, device=DEVICE)
            loss_replay_ce = torch.tensor(0.0, device=DEVICE)
            loss_flux = torch.tensor(0.0, device=DEVICE)

            #
            # Replay
            #
            sample = replay_buffer.sample(2 * len(x))

            if sample is not None:

                rx, ry, rz, rlogits = sample

                rx = rx.to(DEVICE)
                ry = ry.to(DEVICE)
                rz = rz.to(DEVICE)
                rlogits = rlogits.to(DEVICE)

                pred = model(rx)

                #
                # DER++ replay objective
                #
                loss_mse = F.mse_loss(pred, rlogits)

                loss_replay_ce = criterion(pred, ry)

                loss = (
                        loss
                        + loss_mse
                        + der_replay_weight * loss_replay_ce
                )

                #
                # FlowLess regularizer
                #
                if lambda_flux > 0:

                    loss_flux = flux_loss(
                        model=model,
                        x=rx,
                        z_ref=rz,
                        layer=replay_buffer.layer,
                    )
                    loss = loss + lambda_flux * loss_flux

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_task += loss_task.item()
            running_mse += loss_mse.item()
            running_replay_ce += loss_replay_ce.item()
            running_flux += loss_flux.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        logs.append({
            "epoch": epoch,
            "loss": running_loss / n_batches,
            "task_loss": running_task / n_batches,
            "replay_mse": running_mse / n_batches,
            "replay_ce": running_replay_ce / n_batches,
            "flux_loss": running_flux / n_batches,
        })

    return logs


@torch.no_grad()
def evaluate(model, dataset, batch_size):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
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
        der_replay_weight,
        seed,
        out_dir,
):
    set_seed(seed)
    dataset = Dataset(dataset_name)
    cfg = get_experiment_config(dataset_name)
    model = cfg["model_factory"]().to(DEVICE)
    replay_buffer = FluxReplayBuffer(
        memory_per_task=memory_per_task,
        layer=cfg["layer"],
    )

    num_tasks = dataset.num_tasks()
    acc_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

    train_logs = []

    for current_task in range(num_tasks):
        train_taskset, _ = dataset.get_task(current_task)

        print()
        print("=" * 70)
        print(
            f"seed={seed} "
            f"memory={memory_per_task} "
            f"lambda={lambda_flux:g} "
            f"task={current_task}"
        )
        print("=" * 70)

        logs = train_task(
            model=model,
            dataset=train_taskset,
            replay_buffer=replay_buffer,
            lambda_flux=lambda_flux,
            der_replay_weight=der_replay_weight,
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            optimizer_name=cfg["optimizer"],
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
            acc = evaluate(
                model=model,
                dataset=test_taskset,
                batch_size=cfg["eval_batch_size"],
            )
            acc_matrix[current_task, eval_task] = acc
            print(f"eval task {eval_task}: {acc:.2f}")

    final_avg_acc = float(acc_matrix[-1].mean())
    mean_forgetting = forgetting_score(acc_matrix)

    tag = (
        f"seed{seed}_mem{memory_per_task}_"
        f"lambda{lambda_flux:g}"
    )

    np.save(out_dir / f"{tag}_acc_matrix.npy", acc_matrix)

    with open(out_dir / f"{tag}_train_logs.json", "w") as f:
        json.dump(train_logs, f, indent=4)

    return {
        "algorithm": ALGORITHM,
        "dataset": dataset_name,

        # Experiment configuration
        "model": type(model).__name__,
        "representation_layer": cfg["layer"],
        "epochs_per_task": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "eval_batch_size": cfg["eval_batch_size"],
        "optimizer": cfg["optimizer"],
        "learning_rate": cfg["lr"],

        # FlowLess / DER++
        "seed": seed,
        "memory_per_task": memory_per_task,
        "replay_size": len(replay_buffer),
        "der_replay_weight": der_replay_weight,
        "use_flux_reg": int(use_flux_reg),
        "lambda_flux": lambda_flux,

        # Results
        "final_avg_acc": final_avg_acc,
        "mean_forgetting": mean_forgetting,
    }


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--memory_size",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
    )
    parser.add_argument(
        "--lambda_grid",
        type=str,
        default="0,0.03,0.1,0.3,1.0",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
    )
    parser.add_argument(
        "--der_replay_weight",
        type=float,
        default=2.0,
    )


    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT),
    )

    args = parser.parse_args()
    lambda_grid = [
        float(x)
        for x in args.lambda_grid.split(",")
    ]
    memory_size = args.memory_size
    seeds = parse_int_list(args.seeds)
    memory_tag = "random"

    out_dir = (
            Path(args.out)
            / args.dataset
            / "random"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []



    for seed in seeds:

        for lambda_flux in lambda_grid:

            result = run_condition(
                dataset_name=args.dataset,
                memory_per_task=memory_size,
                use_flux_reg=(lambda_flux > 0),
                lambda_flux=lambda_flux,
                der_replay_weight=args.der_replay_weight,
                seed=seed,
                out_dir=out_dir,
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
