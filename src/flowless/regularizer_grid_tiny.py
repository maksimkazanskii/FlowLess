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
from src.models.resnet import ResNet18

import time
def get_experiment_config(dataset_name):

    dataset_name = dataset_name.lower()

    if dataset_name in {"tinyimagenet", "tiny_imagenet"}:
        return {
            "model_factory": lambda: ResNet18(num_classes=200),
            "layer": "layer4",
            "epochs": 40,
            "batch_size": 64,
            "eval_batch_size": 64,
            "lr": 0.1,
            "optimizer": "sgd",
            "replay_weight": 2.0,
        }

    if dataset_name == "mnist":
        return {
            "model_factory": lambda: MLP(),
            "layer": "layer3",
            "epochs": 5,
            "batch_size": 256,
            "eval_batch_size": 512,
            "lr": 1e-3,
            "optimizer": "adam",
            "replay_weight": 2.0,
        }

    if dataset_name in {"fashion_mnist"}:
        return {
            "model_factory": lambda: MLP(),
            "layer": "layer3",
            "epochs": 5,
            "batch_size": 256,
            "eval_batch_size": 512,
            "lr": 1e-3,
            "optimizer": "adam",
            "replay_weight": 2.0,
        }

    if dataset_name in {"cifar10"}:
        return {
            "model_factory": lambda: ResNet18(num_classes=10),
            "layer": "layer4",
            "epochs": 40,
            "batch_size": 128,
            "eval_batch_size": 256,
            "lr": 0.1,
            "optimizer": "sgd",
            "replay_weight": 2.0,
        }

    raise ValueError(
        f"Unsupported dataset: {dataset_name}"
    )
def compute_density(Z, query_Z, k=10):
    if len(Z) <= k:
        return np.ones(len(query_Z))

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(Z)

    distances, _ = nn.kneighbors(query_Z)

    # If query_Z points come directly from Z, first neighbor is itself
    distances = distances[:, 1:]

    rho = k / (distances.sum(axis=1) + 1e-12)

    return rho

OUT = Path("data/results/flowless")

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
#DEVICE = "cpu"

REPLAY_WEIGHT = 2.0
LAMBDA_FLUX = 0.1
lambda_grid = [0.0, 0.1, 0.3, 1.0, 3.0]

# Density weighting:
# alpha=0.0 gives the original unweighted FlowLess loss.
DENSITY_EPS = 1e-12



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
            layer,
            density_weighting=False,
    ):
        self.memory_per_task = memory_per_task
        self.density_weighting = density_weighting

        self.x = []
        self.y = []
        self.z_ref = []
        self.rho = []

        self.layer = layer

    def __len__(self):
        return len(self.x)

    @torch.no_grad()
    def add_dataset(self, model, dataset):

        if self.memory_per_task <= 0:
            return

        model.eval()

        memory_size = min(
            self.memory_per_task,
            len(dataset),
        )

        # --------------------------------------------------
        # Compute representations for the entire current task
        # --------------------------------------------------
        rho = None

        if self.density_weighting:

            # --------------------------------------------------
            # Compute representations for the entire current task
            # --------------------------------------------------
            features = []

            loader = DataLoader(
                dataset,
                batch_size=256,
                shuffle=False,
                num_workers=2,
                pin_memory=False,
                persistent_workers=True,
                prefetch_factor=2,
            )

            features = []

            with torch.no_grad():

                for x,_ in loader:

                    x = x.to(
                        DEVICE,
                        non_blocking=True,
                    )

                    z = model.features(x)[self.layer]
                    z = F.normalize(z, dim=1)

                    features.append(
                        z.cpu()
                    )

            features = torch.cat(features).numpy()

            rho = compute_density(features)

        # --------------------------------------------------
        # Replay selection remains completely random
        # --------------------------------------------------
        idx = np.random.choice(
            len(dataset),
            size=memory_size,
            replace=False,
        )

        # --------------------------------------------------
        # Store samples, reference features, and densities
        # --------------------------------------------------
        xs = torch.stack([dataset[i][0] for i in idx]).to(DEVICE)
        ys = [dataset[i][1] for i in idx]

        with torch.no_grad():
            z_refs = model.features(xs)[self.layer].cpu()

        for sample_idx, x, y, z_ref in zip(idx, xs.cpu(), ys, z_refs):

            self.x.append(x.clone())
            self.y.append(int(y))
            self.z_ref.append(z_ref.clone())

            if self.density_weighting:
                self.rho.append(float(rho[sample_idx]))
    def sample(self, batch_size):
        if len(self.x) == 0:
            return None, None, None, None

        idx = np.random.choice(
            len(self.x),
            min(batch_size, len(self.x)),
            replace=False,
        )

        x = torch.stack([
            self.x[i] for i in idx
        ])

        y = torch.tensor(
            [self.y[i] for i in idx],
            dtype=torch.long,
        )

        z_ref = torch.stack([
            self.z_ref[i] for i in idx
        ])

        if self.density_weighting:

            rho = torch.tensor(
                [self.rho[i] for i in idx],
                dtype=torch.float32,
            )

        else:

            rho = None

        return x, y, z_ref, rho


def flux_loss(
        model,
        x,
        z_ref,
        rho,
        alpha,
        layer,
        normalize=True,
):
    z_now = model.features(x)[layer]

    if normalize:
        z_now = F.normalize(z_now, dim=1)
        z_ref = F.normalize(z_ref, dim=1)

    # Squared representation movement for each sample
    displacement_sq = (
            (z_now - z_ref) ** 2
    ).sum(dim=1)

    # Inverse-density weighting:
    # sparse samples receive larger weights
    if rho is None or alpha == 0.0:
        return displacement_sq.mean()

    weights = (rho + DENSITY_EPS).pow(-alpha)
    weights = weights / (weights.mean() + DENSITY_EPS)

    return (weights * displacement_sq).mean()


def train_task(
        model,
        dataset,
        replay_buffer,
        use_flux_reg,
        lambda_flux,
        alpha,
        batch_size,
        epochs,
        lr,
        optimizer_name,
        replay_weight,
):
    #print("[T1] Creating DataLoader", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    #print("[T2] DataLoader created", flush=True)

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
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )

    #print("[T3] Optimizer created", flush=True)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    #print("[T4] GradScaler created", flush=True)

    scheduler = None

    if optimizer_name == "sgd":
        warmup_epochs = 5

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    #print("[T5] Scheduler created", flush=True)

    criterion = nn.CrossEntropyLoss()
    model.train()

    logs = []

    #print("[T6] Creating loader iterator", flush=True)
    loader_iterator = iter(loader)
    #print("[T7] Loader iterator created", flush=True)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        running_correct = 0
        running_total = 0

        running_loss = 0.0
        running_task = 0.0
        running_replay = 0.0
        running_flux = 0.0
        n_batches = 0

        #print(
        #    f"[T8] Starting epoch {epoch + 1}/{epochs}",
        #    flush=True,
        #)

        for batch_idx, (x, y) in enumerate(loader):


            x = x.to(DEVICE)
            y = y.to(DEVICE)



            rx, ry, rz, rrho = replay_buffer.sample(
                2 * len(x)
            )

            if rx is not None:
                rx = rx.to(DEVICE)
                ry = ry.to(DEVICE)
                rz = rz.to(DEVICE)

                if rrho is not None:
                    rrho = rrho.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)



            with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(DEVICE.type == "cuda"),
            ):
                logits = model(x)



                loss_task = criterion(logits, y)
                loss = loss_task

                pred = logits.argmax(dim=1)

                running_correct += (
                        pred == y
                ).sum().item()

                running_total += y.size(0)

                loss_replay = torch.zeros(
                    (),
                    device=DEVICE,
                )

                loss_flux = torch.zeros(
                    (),
                    device=DEVICE,
                )

                if rx is not None:
                    replay_logits = model(rx)

                    loss_replay = criterion(
                        replay_logits,
                        ry,
                    )

                    loss = (
                            loss
                            + replay_weight * loss_replay
                    )

                    if use_flux_reg:
                        loss_flux = flux_loss(
                            model=model,
                            x=rx,
                            z_ref=rz,
                            rho=rrho,
                            alpha=(
                                alpha
                                if replay_buffer.density_weighting
                                else 0.0
                            ),
                            layer=replay_buffer.layer,
                        )

                        loss = (
                                loss
                                + lambda_flux * loss_flux
                        )



            scaler.scale(loss).backward()


            scaler.step(optimizer)
            scaler.update()



            running_loss += float(
                loss.detach().item()
            )

            running_task += float(
                loss_task.detach().item()
            )

            running_replay += float(
                loss_replay.detach().item()
            )

            running_flux += float(
                loss_flux.detach().item()
            )

            n_batches += 1



        if scheduler is not None:
            scheduler.step()

        epoch_time = (
                time.perf_counter() - epoch_start
        )

        logs.append({
            "epoch": epoch,
            "epoch_time": epoch_time,
            "loss": running_loss / max(n_batches, 1),
            "task_loss": running_task / max(n_batches, 1),
            "replay_loss": running_replay / max(n_batches, 1),
            "flux_loss": running_flux / max(n_batches, 1),
        })

        epoch_acc = (
                100.0
                * running_correct
                / max(running_total, 1)
        )

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Time: {epoch_time:.1f}s | "
            f"Acc: {epoch_acc:.2f}% | "
            f"Loss: "
            f"{running_loss / max(n_batches, 1):.4f} | "
            f"Task: "
            f"{running_task / max(n_batches, 1):.4f} | "
            f"Replay: "
            f"{running_replay / max(n_batches, 1):.4f} | "
            f"Flux: "
            f"{running_flux / max(n_batches, 1):.4f}",
            flush=True,
        )

    return logs


@torch.no_grad()
def evaluate(
        model,
        dataset,
        batch_size,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
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
        seed,
        out_dir,
        density_weighting,
        alpha,
):
    set_seed(seed)


    cfg = get_experiment_config(dataset_name)


    dataset = Dataset(dataset_name)

    model = cfg["model_factory"]().to(DEVICE)

    replay_buffer = FluxReplayBuffer(
        memory_per_task=memory_per_task,
        layer=cfg["layer"],
        density_weighting=density_weighting,
    )

    num_tasks = dataset.num_tasks()


    acc_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)
    train_logs = []

    for current_task in range(num_tasks):


        train_taskset, _ = dataset.get_task(current_task)



        x0, y0 = train_taskset[0]



        logs = train_task(
            model=model,
            dataset=train_taskset,
            replay_buffer=replay_buffer,
            use_flux_reg=use_flux_reg,
            lambda_flux=lambda_flux,
            alpha=alpha,
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            optimizer_name=cfg["optimizer"],
            replay_weight=cfg["replay_weight"],
        )

        print("[15] Training finished")

        for row in logs:
            row.update({
                "seed": seed,
                "memory_per_task": memory_per_task,
                "use_flux_reg": int(use_flux_reg),
                "task": current_task,
                "lambda_flux": lambda_flux,
                "alpha": alpha,
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



    final_avg_acc = float(acc_matrix[-1].mean())
    mean_forgetting = forgetting_score(acc_matrix)

    tag = (
        f"seed{seed}_mem{memory_per_task}_"
        f"flux{int(use_flux_reg)}_"
        f"lambda{lambda_flux:g}_"
        f"alpha{alpha:g}"
    )

    np.save(out_dir / f"{tag}_acc_matrix.npy", acc_matrix)

    with open(out_dir / f"{tag}_train_logs.json", "w") as f:
        json.dump(train_logs, f, indent=4)



    return {
        "dataset": dataset_name,
        "model": model.__class__.__name__,
        "representation_layer": cfg["layer"],
        "epochs_per_task": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "optimizer": cfg["optimizer"],
        "learning_rate": cfg["lr"],
        "seed": seed,
        "density_weighting": int(density_weighting),
        "alpha": alpha,
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
        "--density_weighting",
        action="store_true",
        help="Weight FlowLess by density.",
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Density weighting exponent.",
    )

    args = parser.parse_args()

    memory_sizes = parse_int_list(args.memory_sizes)
    seeds = parse_int_list(args.seeds)
    memory_tag = (
        "density_weighted"
        if args.density_weighting
        else "standard"
    )

    if args.density_weighting:
        out_dir = (
                Path(args.out)
                / args.dataset
                / memory_tag
                / f"alpha_{args.alpha:g}"
        )
    else:
        out_dir = (
            Path(args.out)
            / args.dataset
            / memory_tag
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []



    for seed in seeds:
        for memory_per_task in memory_sizes:
            for lambda_flux in lambda_grid:

                tag = (
                    f"seed{seed}_mem{memory_per_task}_"
                    f"flux{int(lambda_flux > 0)}_"
                    f"lambda{lambda_flux:g}_"
                    f"alpha{args.alpha:g}"
                )

                acc_file = out_dir / f"{tag}_acc_matrix.npy"

                if acc_file.exists():
                    print(f"[SKIP] {tag}")
                    continue

                result = run_condition(
                    dataset_name=args.dataset,
                    memory_per_task=memory_per_task,
                    use_flux_reg=(lambda_flux > 0),
                    lambda_flux=lambda_flux,
                    seed=seed,
                    out_dir=out_dir,
                    density_weighting=args.density_weighting,
                    alpha=args.alpha,
                )

                results.append(result)

                if args.dataset.lower() in {
                    "cifar10",
                    "tinyimagenet",
                    "tiny_imagenet",
                }:
                    results_csv = out_dir / f"results_seed{seed}.csv"
                else:
                    results_csv = out_dir / "results.csv"

                write_header = not results_csv.exists()

                with open(results_csv, "a", newline="") as f:

                    writer = csv.DictWriter(
                        f,
                        fieldnames=list(result.keys()),
                    )

                    if write_header:
                        writer.writeheader()

                    writer.writerow(result)

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
