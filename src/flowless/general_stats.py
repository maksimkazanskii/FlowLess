import argparse
from pathlib import Path

import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.results)

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

    out_dir = Path(args.results).parent

    summary.to_csv(
        out_dir / "derpp_lambda_sweep_summary.csv",
        index=False,
        )

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
