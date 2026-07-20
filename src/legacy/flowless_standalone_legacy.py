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





OUT = Path("data/results/flowless")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3







def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StandaloneFlowLess:

    def __init__(
            self,
            layer="layer3",
            decay=1.0,
            normalize=True,
    ):
        self.layer = layer
        self.decay = decay
        self.normalize = normalize

        self.theta_ref = {}
        self.importance = {}

    def __len__(self):
        return len(self.importance)

    @torch.no_grad()
    def snapshot(self, model):
        """
        Save parameters after finishing a task.
        """

        self.theta_ref = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    def penalty(self, model):

        if len(self.theta_ref) == 0:
            return torch.tensor(
                0.0,
                device=next(model.parameters()).device,
            )

        loss = torch.tensor(
            0.0,
            device=next(model.parameters()).device,
        )

        for name, p in model.named_parameters():

            if name not in self.importance:
                continue

            loss += (
                    self.importance[name]
                    * (p - self.theta_ref[name]).pow(2)
            ).sum()

        return loss

    def estimate_importance(
            self,
            model,
            dataset,
            batch_size=256,
    ):
        """
        Estimate diagonal Jacobian importance
        Ω_i = E ||∂z/∂θ_i||²
        """

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        model.eval()

        importance = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        n_batches = 0

        num_projections = 4

        for x, _ in loader:
            x = x.to(DEVICE)
            z = model.features(x)[self.layer]


            for proj in range(num_projections):
                model.zero_grad(set_to_none=True)

                v = torch.empty_like(z).bernoulli_(0.5).mul_(2).sub_(1)

                scalar = (z * v).sum() / np.sqrt(z.numel())

                scalar.backward(
                    retain_graph=(proj < num_projections - 1)
                )

                for name, p in model.named_parameters():
                    if p.grad is not None:
                        importance[name] += p.grad.detach().pow(2)

                n_batches += 1

        for name in importance:

            importance[name] /= max(n_batches, 1)

            if self.normalize:

                importance[name] /= (
                        importance[name].mean() + 1e-12
                )

        #
        # online accumulation
        #
        if len(self.importance) == 0:

            self.importance = importance

        else:

            for name in importance:

                self.importance[name] = (
                        self.decay * self.importance[name]
                        + importance[name]
                )

        self.snapshot(model)

        #
        # diagnostics
        #
        values = torch.cat([
            v.flatten()
            for v in self.importance.values()
        ])

        print(
            f"importance: "
            f"mean={values.mean():.4f}, "
            f"std={values.std():.4f}, "
            f"max={values.max():.4f}"
        )




def train_task(
    model,
    dataset,
    flowless,
    use_flowless,
    lambda_flowless,
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
        running_flowless = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss_task = criterion(logits, y)
            loss_flowless = loss_task.new_tensor(0.0)

            if use_flowless:
                loss_flowless = flowless.penalty(model)

            loss = loss_task + lambda_flowless * loss_flowless


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_task += float(loss_task.item())
            running_flowless += float(loss_flowless.item())
            n_batches += 1

        logs.append({
            "epoch": epoch,
            "loss": running_loss / max(n_batches, 1),
            "task_loss": running_task / max(n_batches, 1),
            "flowless_loss": running_flowless / max(n_batches, 1),
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
        lambda_flowless,
        seed,
        layer,
        out_dir,
):
    set_seed(seed)
    dataset = Dataset(dataset_name)
    model = MLP().to(DEVICE)
    flowless = StandaloneFlowLess(layer=layer)

    num_tasks = dataset.num_tasks()
    acc_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

    train_logs = []

    for current_task in range(num_tasks):
        train_taskset, _ = dataset.get_task(current_task)

        print()
        print("=" * 70)
        print(
            f"seed={seed} "
            f"lambda={lambda_flowless:g} "
            f"task={current_task}"
        )
        print("=" * 70)

        logs = train_task(
            model=model,
            dataset=train_taskset,
            flowless=flowless,
            use_flowless=(lambda_flowless > 0),
            lambda_flowless=lambda_flowless,
        )

        for row in logs:
            row.update({
                "seed": seed,
                "task": current_task,
                "lambda_flowless": lambda_flowless,
            })

        train_logs.extend(logs)

        if lambda_flowless > 0:
            flowless.estimate_importance(
                model,
                train_taskset,
            )

        for eval_task in range(current_task + 1):
            _, test_taskset = dataset.get_task(eval_task)
            acc = evaluate(model, test_taskset)
            acc_matrix[current_task, eval_task] = acc
            print(f"eval task {eval_task}: {acc:.2f}")

    final_avg_acc = float(acc_matrix[-1].mean())
    mean_forgetting = forgetting_score(acc_matrix)

    tag = (
        f"seed{seed}"
        f"_lambda{lambda_flowless:g}"
        f"_layer{layer}"
    )

    np.save(out_dir / f"{tag}_acc_matrix.npy", acc_matrix)

    with open(out_dir / f"{tag}_train_logs.json", "w") as f:
        json.dump(train_logs, f, indent=4)

    return {
        "dataset": dataset_name,
        "seed": seed,
        "lambda_flowless": lambda_flowless,
        "layer": layer,
        "final_avg_acc": final_avg_acc,
        "mean_forgetting": mean_forgetting,
    }


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
    )

    parser.add_argument(
        "--layer",
        default="layer3",
    )

    parser.add_argument(
        "--lambda-grid",
        default="0,0.1,0.3,1,3",
    )


    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT),
    )

    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)

    lambda_grid = [
        float(x)
        for x in args.lambda_grid.split(",")
        if x.strip()
    ]



    out_dir = (
            Path(args.out)
            / args.dataset
            / "standalone"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []



    for seed in seeds:
        for lambda_flowless in lambda_grid:
            result = run_condition(
                dataset_name=args.dataset,
                lambda_flowless=lambda_flowless,
                seed=seed,
                layer=args.layer,
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
