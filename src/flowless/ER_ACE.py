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



ALGORITHM = "ER-ACE"


OUT = Path("data/results/flowless_erace")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3


LAYER = "layer3"



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FluxReplayBuffer:

    def __init__(self, memory_per_task):
        self.memory_per_task = memory_per_task

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
                model.features(xb)[LAYER]
                .squeeze(0)
                .detach()
                .cpu()
            )

            self.x.append(x.clone())
            self.y.append(int(y))
            self.z_ref.append(z.clone())

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

        return x, y, z_ref


def flux_loss(model, x, z_ref, normalize=True):
    z_now = model.features(x)[LAYER]

    if normalize:
        z_now = F.normalize(z_now, dim=1)
        z_ref = F.normalize(z_ref, dim=1)

    return ((z_now - z_ref) ** 2).sum(dim=1).mean()


def train_task(
        model,
        dataset,
        replay_buffer,
        lambda_flux,
        replay_weight,
        seen_classes,
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
            mask = torch.zeros(
                logits.size(1),
                dtype=torch.bool,
                device=DEVICE,
            )

            allowed = set(seen_classes)
            allowed.update(torch.unique(y).tolist())

            mask[list(allowed)] = True

            masked_logits = logits.clone()
            masked_logits[:, ~mask] = -1e9

            loss_task = criterion(masked_logits, y)

            loss = loss_task

            #
            # Initialize logging variables
            #

            loss_replay_ce = torch.tensor(0.0, device=DEVICE)
            loss_flux = torch.tensor(0.0, device=DEVICE)

            #
            # ER
            #
            sample = replay_buffer.sample(2 * len(x))

            if sample is not None:

                rx, ry, rz = sample

                rx = rx.to(DEVICE)
                ry = ry.to(DEVICE)
                rz = rz.to(DEVICE)


                pred = model(rx)




                loss_replay_ce = criterion(pred, ry)

                loss = (
                        loss
                        + replay_weight * loss_replay_ce
                )

                #
                # FlowLess regularizer
                #
                if lambda_flux > 0:

                    loss_flux = flux_loss(
                        model,
                        rx,
                        rz,
                    )

                    loss = loss + lambda_flux * loss_flux

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_task += loss_task.item()

            running_replay_ce += loss_replay_ce.item()
            running_flux += loss_flux.item()
            n_batches += 1

        logs.append({
            "epoch": epoch,
            "loss": running_loss / n_batches,
            "task_loss": running_task / n_batches,
            "replay_ce": running_replay_ce / n_batches,
            "flux_loss": running_flux / n_batches,
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
        replay_weight,
        seed,
        out_dir,
):
    set_seed(seed)
    dataset = Dataset(dataset_name)
    model = MLP().to(DEVICE)
    replay_buffer = FluxReplayBuffer(memory_per_task)
    seen_classes = set()

    num_tasks = dataset.num_tasks()
    acc_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

    train_logs = []

    for current_task in range(num_tasks):
        train_taskset, _ = dataset.get_task(current_task)
        labels = [
            train_taskset.dataset.targets[i]
            for i in train_taskset.indices
        ]

        if torch.is_tensor(labels):
            labels = labels.tolist()

        seen_classes.update(int(c) for c in labels)
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
            replay_weight=replay_weight,
            seen_classes=seen_classes,
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
        "seed": seed,
        "memory_per_task": memory_per_task,
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
        "--replay_weight",
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
                replay_weight=args.replay_weight,
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
