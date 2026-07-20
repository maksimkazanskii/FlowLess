import argparse
from pathlib import Path
import pandas as pd

REQUIRED = {
    "seed",
    "memory_per_task",
    "lambda_flux",
    "final_avg_acc",
    "mean_forgetting",
}

def load_all(root: Path):
    if root.is_file():
        files = [root]
    else:
        files = sorted(root.rglob("results.csv"))

    if not files:
        raise FileNotFoundError("No results.csv files found.")

    dfs = []
    for f in files:
        df = pd.read_csv(f)

        missing = REQUIRED - set(df.columns)

        if missing:
            print(f"Skipping {f} (missing {sorted(missing)})")
            continue

        if "alpha" not in df.columns:
            alpha = 0.0
            for p in f.parts:
                if p.startswith("alpha_"):
                    try:
                        alpha = float(p.replace("alpha_", ""))
                    except Exception:
                        pass
            df["alpha"] = alpha

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Directory or results.csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.results)
    out = Path(args.out) if args.out else (
        root.parent if root.is_file() else root
    )
    out.mkdir(parents=True, exist_ok=True)

    df = load_all(root)

    stats = (
        df.groupby(
            ["alpha", "memory_per_task", "lambda_flux"]
        )
        .agg(
            final_avg_acc_mean=("final_avg_acc", "mean"),
            final_avg_acc_std=("final_avg_acc", "std"),
            mean_forgetting_mean=("mean_forgetting", "mean"),
            mean_forgetting_std=("mean_forgetting", "std"),
            n=("seed", "count"),
        )
        .reset_index()
        .fillna(0.0)
        .sort_values(
            ["alpha", "memory_per_task", "lambda_flux"]
        )
    )

    # -------------------------------------------------------
    # Best lambda for every alpha and memory
    # -------------------------------------------------------
    best = (
        stats
        .sort_values(
            [
                "alpha",
                "memory_per_task",
                "final_avg_acc_mean",
            ],
            ascending=[True, True, False],
        )
        .groupby(
            ["alpha", "memory_per_task"],
            as_index=False,
        )
        .head(1)
    )

    best.to_csv(
        out / "best_lambda_by_alpha.csv",
        index=False,
        )

    print("\n" + "=" * 90)
    print("BEST LAMBDA PER ALPHA")
    print("=" * 90)
    print(best.to_string(index=False))

    # -------------------------------------------------------
    # Alpha summary (using best lambda only)
    # -------------------------------------------------------
    alpha_summary = (
        best
        .groupby("alpha")
        .agg(
            final_avg_acc_mean=("final_avg_acc_mean", "mean"),
            final_avg_acc_std=("final_avg_acc_mean", "std"),
            mean_forgetting_mean=("mean_forgetting_mean", "mean"),
            mean_forgetting_std=("mean_forgetting_mean", "std"),
            mean_best_lambda=("lambda_flux", "mean"),
            n_memory=("memory_per_task", "count"),
        )
        .reset_index()
        .fillna(0.0)
        .sort_values("alpha")
    )

    alpha_summary.to_csv(
        out / "alpha_summary.csv",
        index=False,
        )

    print("\n" + "=" * 90)
    print("ALPHA SUMMARY")
    print("=" * 90)
    print(alpha_summary.to_string(index=False))

    stats.to_csv(out / "full_stats.csv", index=False)
    alpha_summary.to_csv(out / "alpha_summary.csv", index=False)

    print("=" * 90)
    print("FULL STATS")
    print("=" * 90)
    print(stats.to_string(index=False))

    print()
    print("=" * 90)
    print("ALPHA SUMMARY")
    print("=" * 90)
    print(alpha_summary.to_string(index=False))

    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()


# mnist

# density_weighted
#python src/flowless/regularizer_grid_stats_alpha.py \
#    --results data/results/flowless/mnist/standard or density_weighted


