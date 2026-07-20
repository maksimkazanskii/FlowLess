import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

ROOT = Path("data/mnist/mlp/replay")
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

LAYERS = ["layer1", "layer2", "layer3"]
METRICS = ["effective_dim", "density_mean", "density_std"]

def compute_density(Z, k=10):

    nn = NearestNeighbors(
        n_neighbors=k + 1
    )

    nn.fit(Z)

    distances, _ = nn.kneighbors(Z)

    distances = distances[:, 1:]

    rho = k / (
            distances.sum(axis=1)
            + 1e-12
    )

    return rho

def plot_density_evolution(
        snapshot_dirs,
        probe_id,
        layer=LAYER
):

    plt.figure(figsize=(8, 6))

    for snapshot in snapshot_dirs:

        Z = load_representation(
            snapshot,
            probe_id,
            layer
        )

        if Z is None:
            continue

        rho = compute_density(Z)

        plt.hist(
            rho,
            bins=50,
            density=True,
            alpha=0.25,
            label=snapshot.name
        )

    plt.xlabel("Density")
    plt.ylabel("Probability")

    plt.title(
        f"Task {probe_id} Density Evolution"
    )

    plt.legend(
        fontsize=6,
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        OUT /
        f"task_{probe_id}_density_evolution.png",
        dpi=300
    )

    plt.close()

def get_snapshot_index(name):

    if name == "snapshot_init":
        return -1

    parts = name.split("_")

    # old format
    if len(parts) == 2:
        return int(parts[1])

    # new format
    task = int(parts[1].replace("task", ""))
    epoch = int(parts[2].replace("epoch", ""))

    return task * 5 + epoch


def load_metric(metric, layer):
    curves = {}

    snapshot_dirs = sorted(
        ROOT.glob("snapshot_*"),
        key=lambda x: get_snapshot_index(x.name)
    )

    for snapshot_dir in snapshot_dirs:

        snap_idx = get_snapshot_index(
            snapshot_dir.name
        )

        probe_dirs = sorted(
            snapshot_dir.glob(
                "probe_task_*"
            )
        )

        for probe_dir in probe_dirs:

            probe_id = int(
                probe_dir.name.split("_")[-1]
            )

            stats_file = (
                    probe_dir /
                    "stats.json"
            )

            if not stats_file.exists():
                continue

            with open(stats_file) as f:
                stats = json.load(f)

            value = stats[layer][metric]

            if probe_id not in curves:
                curves[probe_id] = {
                    "x": [],
                    "y": []
                }

            curves[probe_id]["x"].append(
                snap_idx
            )

            curves[probe_id]["y"].append(
                value
            )

    return curves


def make_metric_plot(metric, layer):

    curves = load_metric(
        metric,
        layer
    )

    if len(curves) == 0:
        return

    plt.figure(figsize=(8, 5))

    for probe_id in sorted(curves):

        plt.plot(
            curves[probe_id]["x"],
            curves[probe_id]["y"],
            marker="o",
            label=f"Task {probe_id + 1}"
        )

    plt.xlabel("Snapshot")
    plt.ylabel(metric)
    plt.title(
        f"{metric} - {layer}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS /
        f"{metric}_{layer}.png",
        dpi=300
    )

    plt.close()


def compute_drift(z1, z2):
    return (
            np.linalg.norm(z2 - z1)
            / np.sqrt(z1.size)
    )


def load_representation(
        snapshot_dir,
        probe_id,
        layer
):
    path = (
            snapshot_dir /
            f"probe_task_{probe_id}" /
            f"{layer}.npy"
    )

    if not path.exists():
        return None

    return np.load(path)


def make_drift_plot(layer):

    snapshot_dirs = sorted(
        ROOT.glob("snapshot_*"),
        key=lambda x: get_snapshot_index(x.name)
    )

    max_probe = 0

    for s in snapshot_dirs:
        for p in s.glob("probe_task_*"):
            pid = int(
                p.name.split("_")[-1]
            )
            max_probe = max(
                max_probe,
                pid
            )

    plt.figure(figsize=(8, 5))

    for probe_id in range(
            max_probe + 1
    ):

        xs = []
        ys = []

        prev_Z = None

        for snapshot_dir in snapshot_dirs:

            Z = load_representation(
                snapshot_dir,
                probe_id,
                layer
            )

            if Z is None:
                continue

            snap_idx = get_snapshot_index(
                snapshot_dir.name
            )

            if prev_Z is not None:

                drift = compute_drift(
                    prev_Z,
                    Z
                )

                xs.append(snap_idx)
                ys.append(drift)

            prev_Z = Z

        if len(xs) > 0:

            plt.plot(
                xs,
                ys,
                marker="o",
                label=f"Task {probe_id}"
            )

    plt.xlabel("Snapshot")
    plt.ylabel("Representation Drift")

    plt.title(
        f"Drift - {layer}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS /
        f"drift_{layer}.png",
        dpi=300
    )

    plt.close()


def main():

    for layer in LAYERS:

        for metric in METRICS:

            make_metric_plot(
                metric,
                layer
            )

        make_drift_plot(
            layer
        )

    print(
        f"Plots saved to {PLOTS}"
    )


if __name__ == "__main__":
    main()