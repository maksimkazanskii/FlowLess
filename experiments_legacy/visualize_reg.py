import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("data/mnist/mlp/flux_reg_grid")
OUT = ROOT / "figures"

OUT.mkdir(parents=True, exist_ok=True)


def load_results(root):
    path = root / "results.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    df = pd.read_csv(path)

    required = {
        "seed",
        "memory_per_task",
        "use_flux_reg",
        "lambda_flux",
        "final_avg_acc",
        "mean_forgetting",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in results.csv: {missing}")

    df["method"] = np.where(
        df["use_flux_reg"].astype(int) == 1,
        "Replay + FluxReg",
        "Replay",
    )

    return df


def aggregate(df):
    return (
        df
        .groupby(["memory_per_task", "method"])
        .agg(
            final_avg_acc_mean=("final_avg_acc", "mean"),
            final_avg_acc_sem=(
                "final_avg_acc",
                lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
            ),
            mean_forgetting_mean=("mean_forgetting", "mean"),
            mean_forgetting_sem=(
                "mean_forgetting",
                lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
            ),
            n=("seed", "count"),
        )
        .reset_index()
    )


def plot_metric(
        agg,
        metric_mean,
        metric_sem,
        ylabel,
        filename,
):
    plt.figure(figsize=(6.5, 4.0))

    for method, g in agg.groupby("method"):
        g = g.sort_values("memory_per_task")

        plt.errorbar(
            g["memory_per_task"],
            g[metric_mean],
            yerr=g[metric_sem],
            marker="o",
            linewidth=2.0,
            markersize=5,
            capsize=4,
            label=method,
        )

    plt.xlabel("Replay memory per task")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs replay memory")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    png = OUT / filename
    pdf = OUT / filename.replace(".png", ".pdf")

    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()

    print(f"saved: {png}")
    print(f"saved: {pdf}")


def plot_improvement(df):
    pivot = df.pivot_table(
        index=["seed", "memory_per_task"],
        columns="method",
        values=["final_avg_acc", "mean_forgetting"],
    )

    rows = []

    for idx, row in pivot.iterrows():
        seed, mem = idx

        if (
                ("final_avg_acc", "Replay") not in row
                or ("final_avg_acc", "Replay + FluxReg") not in row
        ):
            continue

        acc_gain = (
            row[("final_avg_acc", "Replay + FluxReg")]
            - row[("final_avg_acc", "Replay")]
        )

        forget_reduction = (
            row[("mean_forgetting", "Replay")]
            - row[("mean_forgetting", "Replay + FluxReg")]
        )

        rows.append({
            "seed": seed,
            "memory_per_task": mem,
            "acc_gain": acc_gain,
            "forgetting_reduction": forget_reduction,
        })

    gains = pd.DataFrame(rows)

    if gains.empty:
        return

    gains_csv = OUT / "fluxreg_improvement.csv"
    gains.to_csv(gains_csv, index=False)

    print(f"saved: {gains_csv}")

    summary = (
        gains
        .groupby("memory_per_task")
        .agg(
            acc_gain_mean=("acc_gain", "mean"),
            acc_gain_sem=(
                "acc_gain",
                lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
            ),
            forgetting_reduction_mean=("forgetting_reduction", "mean"),
            forgetting_reduction_sem=(
                "forgetting_reduction",
                lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
            ),
        )
        .reset_index()
    )

    for col, err, ylabel, filename in [
        (
            "acc_gain_mean",
            "acc_gain_sem",
            "Accuracy gain from FluxReg",
            "fluxreg_accuracy_gain.png",
        ),
        (
            "forgetting_reduction_mean",
            "forgetting_reduction_sem",
            "Forgetting reduction from FluxReg",
            "fluxreg_forgetting_reduction.png",
        ),
    ]:
        plt.figure(figsize=(6.5, 4.0))

        plt.errorbar(
            summary["memory_per_task"],
            summary[col],
            yerr=summary[err],
            marker="o",
            linewidth=2.0,
            markersize=5,
            capsize=4,
        )

        plt.axhline(
            0.0,
            linewidth=1.0,
            alpha=0.5,
        )

        plt.xlabel("Replay memory per task")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        png = OUT / filename
        pdf = OUT / filename.replace(".png", ".pdf")

        plt.savefig(png, dpi=300)
        plt.savefig(pdf)
        plt.close()

        print(f"saved: {png}")
        print(f"saved: {pdf}")


def plot_accuracy_matrices(root):
    matrix_files = sorted(root.glob("*_acc_matrix.npy"))

    for path in matrix_files:
        M = np.load(path)

        plt.figure(figsize=(4.8, 4.2))
        plt.imshow(M, vmin=0, vmax=100, aspect="auto")
        plt.colorbar(label="Accuracy (%)")

        plt.xlabel("Evaluated task")
        plt.ylabel("Training task")
        plt.title(path.stem.replace("_acc_matrix", ""))
        plt.tight_layout()

        png = OUT / f"{path.stem}.png"
        pdf = OUT / f"{path.stem}.pdf"

        plt.savefig(png, dpi=300)
        plt.savefig(pdf)
        plt.close()

        print(f"saved: {png}")
        print(f"saved: {pdf}")


def make_summary_table(agg):
    table = agg.copy()

    table["final_avg_acc"] = (
        table["final_avg_acc_mean"].round(2).astype(str)
        + " ± "
        + table["final_avg_acc_sem"].round(2).astype(str)
    )

    table["mean_forgetting"] = (
        table["mean_forgetting_mean"].round(2).astype(str)
        + " ± "
        + table["mean_forgetting_sem"].round(2).astype(str)
    )

    table = table[
        [
            "memory_per_task",
            "method",
            "final_avg_acc",
            "mean_forgetting",
            "n",
        ]
    ]

    path = OUT / "summary_table.csv"
    table.to_csv(path, index=False)

    print(f"saved: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default=str(ROOT),
    )

    parser.add_argument(
        "--matrices",
        action="store_true",
    )

    args = parser.parse_args()

    global ROOT, OUT

    ROOT = Path(args.root)
    OUT = ROOT / "figures"

    OUT.mkdir(parents=True, exist_ok=True)

    df = load_results(ROOT)
    agg = aggregate(df)

    agg_csv = OUT / "aggregated_results.csv"
    agg.to_csv(agg_csv, index=False)

    print(f"saved: {agg_csv}")

    make_summary_table(agg)

    plot_metric(
        agg,
        metric_mean="final_avg_acc_mean",
        metric_sem="final_avg_acc_sem",
        ylabel="Final average accuracy (%)",
        filename="final_average_accuracy.png",
    )

    plot_metric(
        agg,
        metric_mean="mean_forgetting_mean",
        metric_sem="mean_forgetting_sem",
        ylabel="Mean forgetting (%)",
        filename="mean_forgetting.png",
    )

    plot_improvement(df)

    if args.matrices:
        plot_accuracy_matrices(ROOT)

    print(f"done: {OUT}")


if __name__ == "__main__":
    main()
