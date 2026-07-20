#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results.csv",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.results)

    out_dir = Path(args.results).parent

    #
    # Layer summary
    #
    layer_summary = (
        df
        .groupby("layers")
        .agg(
            final_avg_acc_mean=("final_avg_acc", "mean"),
            final_avg_acc_std=("final_avg_acc", "std"),
            mean_forgetting_mean=("mean_forgetting", "mean"),
            mean_forgetting_std=("mean_forgetting", "std"),
            n=("seed", "count"),
        )
        .reset_index()
        .sort_values(
            "final_avg_acc_mean",
            ascending=False,
        )
    )

    layer_summary["final_avg_acc_std"] = (
        layer_summary["final_avg_acc_std"]
        .fillna(0.0)
    )

    layer_summary["mean_forgetting_std"] = (
        layer_summary["mean_forgetting_std"]
        .fillna(0.0)
    )

    layer_summary.to_csv(
        out_dir / "layer_summary.csv",
        index=False,
        )

    #
    # Layer x lambda summary
    #
    layer_lambda_summary = (
        df
        .groupby(
            [
                "layers",
                "lambda_flux",
            ]
        )
        .agg(
            final_avg_acc_mean=("final_avg_acc", "mean"),
            final_avg_acc_std=("final_avg_acc", "std"),
            mean_forgetting_mean=("mean_forgetting", "mean"),
            mean_forgetting_std=("mean_forgetting", "std"),
            n=("seed", "count"),
        )
        .reset_index()
    )

    layer_lambda_summary["final_avg_acc_std"] = (
        layer_lambda_summary["final_avg_acc_std"]
        .fillna(0.0)
    )

    layer_lambda_summary["mean_forgetting_std"] = (
        layer_lambda_summary["mean_forgetting_std"]
        .fillna(0.0)
    )

    layer_lambda_summary.to_csv(
        out_dir / "layer_lambda_summary.csv",
        index=False,
        )

    #
    # Best lambda per layer (accuracy)
    #
    best_lambda_per_layer = (
        layer_lambda_summary
        .sort_values(
            [
                "layers",
                "final_avg_acc_mean",
            ],
            ascending=[True, False],
        )
        .groupby("layers")
        .head(1)
        .reset_index(drop=True)
    )

    best_lambda_per_layer.to_csv(
        out_dir / "best_lambda_per_layer.csv",
        index=False,
        )

    #
    # Best lambda per layer (forgetting)
    #
    best_lambda_per_layer_forgetting = (
        layer_lambda_summary
        .sort_values(
            [
                "layers",
                "mean_forgetting_mean",
            ],
            ascending=[True, True],
        )
        .groupby("layers")
        .head(1)
        .reset_index(drop=True)
    )

    best_lambda_per_layer_forgetting.to_csv(
        out_dir / "best_lambda_per_layer_forgetting.csv",
        index=False,
        )

    #
    # Best layer overall by accuracy
    #
    best_layer_accuracy = (
        layer_summary
        .sort_values(
            "final_avg_acc_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_layer_accuracy.to_csv(
        out_dir / "best_layer_by_accuracy.csv",
        index=False,
        )

    #
    # Best layer overall by forgetting
    #
    best_layer_forgetting = (
        layer_summary
        .sort_values(
            "mean_forgetting_mean",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    best_layer_forgetting.to_csv(
        out_dir / "best_layer_by_forgetting.csv",
        index=False,
        )

    #
    # Overall layer ranking
    #
    overall_layers = (
        df
        .groupby("layers")
        .agg(
            final_avg_acc_mean=("final_avg_acc", "mean"),
            mean_forgetting_mean=("mean_forgetting", "mean"),
            n_runs=("seed", "count"),
        )
        .reset_index()
        .sort_values(
            "final_avg_acc_mean",
            ascending=False,
        )
    )

    overall_layers.to_csv(
        out_dir / "overall_layer_ranking.csv",
        index=False,
        )

    print()
    print("=" * 80)
    print("LAYER SUMMARY")
    print("=" * 80)
    print(layer_summary)

    print()
    print("=" * 80)
    print("BEST LAMBDA PER LAYER (ACCURACY)")
    print("=" * 80)
    print(best_lambda_per_layer)

    print()
    print("=" * 80)
    print("BEST LAMBDA PER LAYER (FORGETTING)")
    print("=" * 80)
    print(best_lambda_per_layer_forgetting)

    print()
    print("=" * 80)
    print("BEST LAYER BY ACCURACY")
    print("=" * 80)
    print(best_layer_accuracy.head())

    print()
    print("=" * 80)
    print("BEST LAYER BY FORGETTING")
    print("=" * 80)
    print(best_layer_forgetting.head())

    print()
    print("=" * 80)
    print("OVERALL LAYER RANKING")
    print("=" * 80)
    print(overall_layers)


if __name__ == "__main__":
    main()