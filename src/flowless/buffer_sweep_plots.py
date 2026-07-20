#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# Global plotting style
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 16,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})

REQUIRED = {
    "memory_per_task",
    "lambda_flux",
    "seed",
    "final_avg_acc",
    "mean_forgetting",
}


def load_results(path: Path):

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("results.csv"))

    if not files:
        raise FileNotFoundError("No results.csv found.")

    dfs = []

    for f in files:

        df = pd.read_csv(f)

        missing = REQUIRED - set(df.columns)

        if missing:
            print(f"Skipping {f}: missing {missing}")
            continue

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def compute_best(df):

    stats = (
        df
        .groupby(
            [
                "memory_per_task",
                "lambda_flux",
            ]
        )
        .agg(
            acc_mean=("final_avg_acc", "mean"),
            acc_std=("final_avg_acc", "std"),
            forget_mean=("mean_forgetting", "mean"),
            forget_std=("mean_forgetting", "std"),
        )
        .reset_index()
    )

    baseline = stats[
        stats.lambda_flux == 0
        ].copy()

    flowless = (
        stats
        .sort_values(
            [
                "memory_per_task",
                "acc_mean",
            ],
            ascending=[True, False],
        )
        .groupby("memory_per_task")
        .head(1)
        .reset_index(drop=True)
    )

    return baseline, flowless


def plot_metric(
        baseline,
        flowless,
        metric,
        ylabel,
        outfile,
):

    plt.figure(figsize=(6, 4))

    x = baseline.memory_per_task

    y = baseline[f"{metric}_mean"]
    s = baseline[f"{metric}_std"]

    plt.plot(
        x,
        y,
        "-o",
        label="ER",
    )

    plt.fill_between(
        x,
        y - s,
        y + s,
        alpha=0.2,
        )

    x = flowless.memory_per_task

    y = flowless[f"{metric}_mean"]
    s = flowless[f"{metric}_std"]

    plt.plot(
        x,
        y,
        "-o",
        label="ER + FlowLess",
    )

    plt.fill_between(
        x,
        y - s,
        y + s,
        alpha=0.2,
        )

    plt.xlabel("Replay buffer size")
    plt.ylabel(ylabel)

    plt.grid(alpha=0.3)

    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(outfile, dpi=300)

    plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    args = parser.parse_args()

    out_dir = Path(
        "data/plots/buffer_sweep"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results(
        Path(args.results)
    )

    baseline, flowless = compute_best(df)

    plot_metric(
        baseline,
        flowless,
        metric="acc",
        ylabel="Final average accuracy (%)",
        outfile=out_dir /
                f"{args.dataset}_accuracy.png",
    )

    plot_metric(
        baseline,
        flowless,
        metric="forget",
        ylabel="Mean forgetting (%)",
        outfile=out_dir /
                f"{args.dataset}_forgetting.png",
    )

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()