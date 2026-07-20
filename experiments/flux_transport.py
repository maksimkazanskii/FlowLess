import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("data/mnist/mlp/replay")
OUT = ROOT / "transport"
LAYER = "layer3"

OUT.mkdir(parents=True, exist_ok=True)


def snapshot_index(path):
    name = path.name

    if name == "snapshot_init":
        return -1

    parts = name.split("_")

    if len(parts) >= 3 and parts[1].startswith("task"):
        task = int(parts[1].replace("task", ""))
        epoch = int(parts[2].replace("epoch", ""))
        return task * 1000 + epoch

    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 10**12


def snapshot_task_epoch(path):
    name = path.name

    if name == "snapshot_init":
        return -1, -1

    parts = name.split("_")

    if len(parts) >= 3 and parts[1].startswith("task"):
        task = int(parts[1].replace("task", ""))
        epoch = int(parts[2].replace("epoch", ""))
        return task, epoch

    return -1, snapshot_index(path)


def load_representation(snapshot, probe_id, layer=LAYER):
    path = snapshot / f"probe_task_{probe_id}" / f"{layer}.npy"
    return np.load(path) if path.exists() else None


def sample_flux(Z_prev, Z_curr):
    if Z_prev.shape != Z_curr.shape:
        raise ValueError(f"Shape mismatch: {Z_prev.shape} vs {Z_curr.shape}")

    disp = np.linalg.norm(Z_curr - Z_prev, axis=1)

    return {
        "flux_mean": float(np.mean(disp)),
        "flux_std": float(np.std(disp)),
        "flux_median": float(np.median(disp)),
        "flux_p90": float(np.percentile(disp, 90)),
        "flux_p95": float(np.percentile(disp, 95)),
        "flux_max": float(np.max(disp)),
    }


def get_snapshots():
    snapshots = [
        p for p in ROOT.iterdir()
        if p.is_dir() and p.name.startswith("snapshot")
    ]

    return sorted(snapshots, key=snapshot_index)


def get_probe_ids(snapshots):
    probe_ids = set()

    for snapshot in snapshots:
        for probe_dir in snapshot.glob("probe_task_*"):
            probe_ids.add(int(probe_dir.name.split("_")[-1]))

    return sorted(probe_ids)


def flux_evolution(snapshots, probe_id, layer=LAYER):
    rows = []

    for prev_snap, curr_snap in zip(snapshots[:-1], snapshots[1:]):
        Z_prev = load_representation(prev_snap, probe_id, layer)
        Z_curr = load_representation(curr_snap, probe_id, layer)

        if Z_prev is None or Z_curr is None:
            continue

        prev_task, prev_epoch = snapshot_task_epoch(prev_snap)
        curr_task, curr_epoch = snapshot_task_epoch(curr_snap)

        stats = sample_flux(Z_prev, Z_curr)

        rows.append({
            "probe": probe_id,
            "layer": layer,
            "snapshot_prev": prev_snap.name,
            "snapshot_curr": curr_snap.name,
            "prev_task": prev_task,
            "prev_epoch": prev_epoch,
            "curr_task": curr_task,
            "curr_epoch": curr_epoch,
            "transition": len(rows),
            **stats,
        })

    return pd.DataFrame(rows)


def plot_flux(df, probe_id):
    if df.empty:
        return

    x = df["transition"].to_numpy()
    y = df["flux_mean"].to_numpy()

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(x, y, marker="o", linewidth=2.0, markersize=5)

    for _, row in df.iterrows():
        if row["prev_task"] != row["curr_task"]:
            plt.axvline(
                row["transition"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.35,
            )

    plt.xlabel("Transition")
    plt.ylabel("Mean representation flux")
    plt.title(f"Probe {probe_id}")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    png = OUT / f"probe_{probe_id}_flux_clean.png"
    pdf = OUT / f"probe_{probe_id}_flux_clean.pdf"

    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()

    print(f"saved: {png}")
    print(f"saved: {pdf}")


def save_probe_flux(snapshots, probe_id):
    df = flux_evolution(snapshots, probe_id)

    csv_path = OUT / f"probe_{probe_id}_flux.csv"
    df.to_csv(csv_path, index=False)

    print(f"saved: {csv_path} rows={len(df)}")

    plot_flux(df, probe_id)

    return df

def plot_all_flux(all_df):
    if all_df.empty:
        return

    plt.figure(figsize=(7.0, 4.2))

    for probe, g in all_df.groupby("probe"):
        plt.plot(
            g["transition"],
            g["flux_mean"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=f"Probe {probe}",
        )

    plt.xlabel("Transition")
    plt.ylabel("Mean representation flux")
    plt.title("Representation flux across probe tasks")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    png = OUT / "all_probe_flux_clean.png"
    pdf = OUT / "all_probe_flux_clean.pdf"

    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()

    print(f"saved: {png}")
    print(f"saved: {pdf}")
def run_flux_analysis():
    snapshots = get_snapshots()

    print(f"ROOT: {ROOT}")
    print(f"snapshots found: {len(snapshots)}")

    if len(snapshots) < 2:
        print("Need at least two snapshots.")
        return

    probe_ids = get_probe_ids(snapshots)

    print(f"probes found: {probe_ids}")

    all_rows = []

    for probe_id in probe_ids:
        df = save_probe_flux(snapshots, probe_id)

        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("No valid probe transitions found.")
        return

    all_df = pd.concat(all_rows, ignore_index=True)

    all_csv = OUT / "all_probe_flux.csv"
    all_df.to_csv(all_csv, index=False)

    summary = (
        all_df
        .groupby("probe")
        .agg(
            flux_mean=("flux_mean", "mean"),
            flux_std=("flux_mean", "std"),
            flux_max=("flux_max", "max"),
            transitions=("transition", "count"),
        )
        .reset_index()
    )

    summary_csv = OUT / "flux_summary.csv"
    summary.to_csv(summary_csv, index=False)

    with open(OUT / "flux_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=4)

    print(f"saved: {all_csv}")
    print(f"saved: {summary_csv}")
    print(f"results saved to: {OUT}")


if __name__ == "__main__":
    run_flux_analysis()