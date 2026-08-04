import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import friedmanchisquare
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel
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
        if (root / "results_seed0.csv").exists():
            files = sorted(root.glob("results_seed*.csv"))
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
    # -------------------------------------------------------
    # Paired t-test against baseline (lambda=0)
    # -------------------------------------------------------

    p_acc = []
    p_forgetting = []

    for _, row in best.iterrows():

        alpha = row["alpha"]
        memory = row["memory_per_task"]
        best_lambda = row["lambda_flux"]

        # baseline
        base = df[
            (df.alpha == alpha) &
            (df.memory_per_task == memory) &
            (df.lambda_flux == 0.0)
            ].sort_values("seed")

        # best FlowLess
        flow = df[
            (df.alpha == alpha) &
            (df.memory_per_task == memory) &
            (df.lambda_flux == best_lambda)
            ].sort_values("seed")

        common = sorted(
            set(base.seed).intersection(flow.seed)
        )

        base = (
            base[base.seed.isin(common)]
            .sort_values("seed")
            .reset_index(drop=True)
        )

        flow = (
            flow[flow.seed.isin(common)]
            .sort_values("seed")
            .reset_index(drop=True)
        )

        # Accuracy
        _, p1 = ttest_rel(
            flow.final_avg_acc,
            base.final_avg_acc,
        )

        # Forgetting
        _, p2 = ttest_rel(
            flow.mean_forgetting,
            base.mean_forgetting,
        )

        p_acc.append(p1)
        p_forgetting.append(p2)

    best["p_acc"] = p_acc
    best["p_forgetting"] = p_forgetting
    # -------------------------------------------------------
    # Holm–Bonferroni correction
    # -------------------------------------------------------

    mask = best["p_acc"].notna()
    best.loc[mask, "p_acc_holm"] = multipletests(
        best.loc[mask, "p_acc"],
        method="holm",
    )[1]

    mask = best["p_forgetting"].notna()
    best.loc[mask, "p_forgetting_holm"] = multipletests(
        best.loc[mask, "p_forgetting"],
        method="holm",
    )[1]
    best.to_csv(
        out / "best_lambda_by_alpha.csv",
        index=False,
        )


    print("\n" + "=" * 90)
    print("BEST LAMBDA PER ALPHA")
    print("=" * 90)
    print(
        best.round(
            {
                "final_avg_acc_mean": 2,
                "final_avg_acc_std": 2,
                "mean_forgetting_mean": 2,
                "mean_forgetting_std": 2,
                "p_acc_holm": 4,
                "p_forgetting_holm": 4,
            }
        ).to_string(index=False)
    )

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

    # -------------------------------------------------------
    # Effect of alpha
    # -------------------------------------------------------

    print("\n" + "=" * 90)
    print("EFFECT OF ALPHA")
    print("=" * 90)

    for memory in sorted(best.memory_per_task.unique()):

        print(f"\nMemory per task = {memory}")

        # best lambda for every alpha
        subset = best[best.memory_per_task == memory]

        # reconstruct per-seed results
        seed_tables = []

        for _, row in subset.iterrows():

            alpha = row.alpha
            lam = row.lambda_flux

            values = (
                df[
                    (df.alpha == alpha)
                    & (df.memory_per_task == memory)
                    & (df.lambda_flux == lam)
                    ][["seed", "final_avg_acc"]]
                .rename(columns={"final_avg_acc": alpha})
            )

            seed_tables.append(values)

        merged = seed_tables[0]

        for t in seed_tables[1:]:
            merged = merged.merge(t, on="seed")

        merged = merged.sort_values("seed")

        alpha_cols = [c for c in merged.columns if c != "seed"]

        if len(alpha_cols) < 3:
            print(f"Only {len(alpha_cols)} alpha value(s) found. Skipping Friedman test.")
            continue

        statistic, p = friedmanchisquare(
            *[merged[c] for c in alpha_cols]
        )

        print(f"Overall Friedman p = {p:.6f}")

        # Pairwise vs alpha = 0
        if 0.0 in alpha_cols:

            print("\nPairwise comparisons vs alpha = 0")

            pvals = []
            names = []

            for a in alpha_cols:

                if a == 0.0:
                    continue

                _, pp = ttest_rel(
                    merged[0.0],
                    merged[a]
                )

                pvals.append(pp)
                names.append(a)

            reject, pvals_corr, _, _ = multipletests(
                pvals,
                method="holm"
            )

            for a, p0, pc in zip(names, pvals, pvals_corr):
                print(
                    f"alpha={a:7.3f} "
                    f"raw={p0:.5f} "
                    f"holm={pc:.5f}"
                )
if __name__ == "__main__":
    main()


# mnist

# density_weighted
#python src/flowless/regularizer_grid_stats_alpha.py \
#    --results data/results/flowless/mnist/standard or density_weighted


