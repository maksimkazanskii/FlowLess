import argparse
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel
import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    from pathlib import Path

    path = Path(args.results)

    if path.is_file():
        df = pd.read_csv(path)
        out_dir = path.parent

    elif path.is_dir():

        csvs = sorted(path.glob("results_seed*.csv"))

        if csvs:
            df = pd.concat(
                [pd.read_csv(f) for f in csvs],
                ignore_index=True,
            )
        else:
            single_file = path / "results.csv"

            if not single_file.exists():
                raise FileNotFoundError(
                    f"Expected results_seed*.csv or results.csv in {path}"
                )

            df = pd.read_csv(single_file)

        out_dir = path

    else:
        raise FileNotFoundError(path)

    #
    # Group columns
    #
    group_cols = ["memory_per_task", "lambda_flux"]

    if "der_replay_weight" in df.columns:
        group_cols.append("der_replay_weight")

    summary = (
        df
        .groupby(group_cols)
        .agg(
            final_avg_acc_mean=("final_avg_acc", "mean"),
            final_avg_acc_std=("final_avg_acc", "std"),
            mean_forgetting_mean=("mean_forgetting", "mean"),
            mean_forgetting_std=("mean_forgetting", "std"),
            n=("seed", "count"),
        )
        .reset_index()
    )

    summary = summary.fillna(0.0)



    #
    # Paired p-values for every lambda vs baseline (lambda=0)
    #
    p_acc = []
    p_forgetting = []

    for _, row in summary.iterrows():

        mem = row["memory_per_task"]
        lam = row["lambda_flux"]

        if lam == 0.0:
            p_acc.append(float("nan"))
            p_forgetting.append(float("nan"))
            continue

        base = (
            df[
                (df.memory_per_task == mem)
                & (df.lambda_flux == 0.0)
                ]
            .sort_values("seed")
        )

        flow = (
            df[
                (df.memory_per_task == mem)
                & (df.lambda_flux == lam)
                ]
            .sort_values("seed")
        )

        merged = base.merge(
            flow,
            on="seed",
            suffixes=("_base", "_flow"),
        )
        # Not enough paired observations for a paired t-test
        if merged["seed"].nunique() < 2:
            p_acc.append(float("nan"))
            p_forgetting.append(float("nan"))
            continue
        _, p1 = ttest_rel(
            merged.final_avg_acc_flow,
            merged.final_avg_acc_base,
        )

        _, p2 = ttest_rel(
            merged.mean_forgetting_flow,
            merged.mean_forgetting_base,
        )

        p_acc.append(p1)
        p_forgetting.append(p2)

    summary["p_acc"] = p_acc
    summary["p_forgetting"] = p_forgetting
    #
    # Multiple-comparison correction (Holm-Bonferroni)
    #

    #
    # Multiple-comparison correction (Holm-Bonferroni)
    #

    summary["p_acc_holm"] = float("nan")
    summary["p_forgetting_holm"] = float("nan")

    mask = summary["p_acc"].notna()

    if mask.any():
        summary.loc[mask, "p_acc_holm"] = multipletests(
            summary.loc[mask, "p_acc"],
            method="holm",
        )[1]

    mask = summary["p_forgetting"].notna()

    if mask.any():
        summary.loc[mask, "p_forgetting_holm"] = multipletests(
            summary.loc[mask, "p_forgetting"],
            method="holm",
        )[1]

#
    # Best accuracy
    #
    best_acc = (
        summary
        .sort_values(
            ["memory_per_task", "final_avg_acc_mean"],
            ascending=[True, False],
        )
        .groupby("memory_per_task")
        .head(1)
        .reset_index(drop=True)
    )

    best_acc.to_csv(
        out_dir / "derpp_best_lambda_by_accuracy.csv",
        index=False,
        )
    #
    # Best forgetting
    #
    best_forgetting = (
        summary
        .sort_values(
            ["memory_per_task", "mean_forgetting_mean"],
            ascending=[True, True],
        )
        .groupby("memory_per_task")
        .head(1)
        .reset_index(drop=True)
    )

    best_forgetting.to_csv(
        out_dir / "derpp_best_lambda_by_forgetting.csv",
        index=False,
        )

    #
    # Overall
    #
    overall_group = ["lambda_flux"]

    if "der_replay_weight" in summary.columns:
        overall_group.append("der_replay_weight")

    overall = (
        summary
        .groupby(overall_group)
        .agg(
            final_avg_acc_mean=("final_avg_acc_mean", "mean"),
            mean_forgetting_mean=("mean_forgetting_mean", "mean"),
            n_conditions=("memory_per_task", "count"),
        )
        .reset_index()
    )

    overall.to_csv(
        out_dir / "derpp_overall_summary.csv",
        index=False,
        )

    print("=" * 80)
    print("LAMBDA SUMMARY")
    print("=" * 80)
    print(summary)

    print()
    print("=" * 80)
    print("BEST LAMBDA (ACCURACY)")
    print("=" * 80)
    print(best_acc)

    print()
    print("=" * 80)
    print("BEST LAMBDA (FORGETTING)")
    print("=" * 80)
    print(best_forgetting)

    print()
    print("=" * 80)
    print("OVERALL")
    print("=" * 80)
    print(overall)


if __name__ == "__main__":
    main()

# RUN:python src/flowless/general_stats.py \
#     --results data/results/flowless_derpp/mnist/random/results.csv
