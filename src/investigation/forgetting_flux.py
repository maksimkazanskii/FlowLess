from pathlib import Path
from scipy.stats import ttest_rel
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import argparse
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument(
    "--root",
    type=Path,
    required=True,
    help="Replay snapshot directory",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Dataset name used as a prefix for all output files",
)

args = parser.parse_args()

ROOT = args.root
DATASET = args.dataset
OUT = ROOT / "forgetting_flux"
LAYER = "layer3"

OUT.mkdir(parents=True, exist_ok=True)

def build_multiseed_summary(seed_dirs):
    seed_summaries = []
    seed_datasets = {}

    for seed_dir in seed_dirs:
        print(f"Building dataset for {seed_dir.name}")

        df_seed = build_dataset(seed_dir)

        if df_seed.empty:
            print(f"Skipped {seed_dir.name}: no data")
            continue

        seed_name = seed_dir.name
        df_seed["seed"] = seed_name
        seed_datasets[seed_name] = df_seed

        summary_seed = summarize(df_seed)
        summary_seed["seed"] = seed_name

        seed_summaries.append(summary_seed)

    if not seed_summaries:
        return pd.DataFrame(), {}

    combined_summary = pd.concat(
        seed_summaries,
        ignore_index=True,
    )

    return combined_summary, seed_datasets

def aggregate_summary_across_seeds(seed_summary):
    return (
        seed_summary
        .groupby(
            [
                "probe",
                "global_epoch",
                "curr_task",
                "curr_epoch",
            ],
            as_index=False,
        )
        .agg(
            flux_mean=("flux_mean", "mean"),
            flux_std=("flux_mean", "std"),

            forgotten_mean=("forgotten_rate", "mean"),
            forgotten_std=("forgotten_rate", "std"),

            learned_mean=("learned_rate", "mean"),
            learned_std=("learned_rate", "std"),

            margin_drop_mean=("margin_drop", "mean"),
            margin_drop_std=("margin_drop", "std"),

            prob_true_drop_mean=("prob_true_drop", "mean"),
            prob_true_drop_std=("prob_true_drop", "std"),

            n_seeds=("seed", "nunique"),
        )
    )
def plot_probe_summary_across_seeds(
        aggregated,
        probe,
        output_dir,
):
    g = aggregated[
        aggregated["probe"] == probe
        ].copy()

    if g.empty:
        return

    g = g.sort_values("global_epoch")

    x = g["global_epoch"].to_numpy()

    flux_mean = g["flux_mean"].to_numpy()
    flux_std = g["flux_std"].fillna(0.0).to_numpy()

    forgotten_mean = g["forgotten_mean"].to_numpy()
    forgotten_std = g["forgotten_std"].fillna(0.0).to_numpy()

    margin_mean = g["margin_drop_mean"].to_numpy()
    margin_std = g["margin_drop_std"].fillna(0.0).to_numpy()

    colors = {
        "flux": "steelblue",
        "forgetting": "darkorange",
        "margin": "red",
    }

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 6.1),
        sharex=True,
    )

    # ==================================================
    # Representation flux
    # ==================================================
    axes[0].fill_between(
        x,
        flux_mean - flux_std,
        flux_mean + flux_std,
        color=colors["flux"],
        alpha=0.18,
        linewidth=0,
        zorder=1,
        )

    axes[0].plot(
        x,
        flux_mean,
        color=colors["flux"],
        marker="o",
        linewidth=2.0,
        markersize=3.5,
        zorder=3,
    )

    axes[0].set_ylabel(
        "Mean flux",
        fontsize=11,
    )

    # ==================================================
    # Hard forgetting
    # ==================================================
    forgotten_lower = np.clip(
        forgotten_mean - forgotten_std,
        0.0,
        1.0,
        )

    forgotten_upper = np.clip(
        forgotten_mean + forgotten_std,
        0.0,
        1.0,
        )

    axes[1].fill_between(
        x,
        forgotten_lower,
        forgotten_upper,
        color=colors["forgetting"],
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )

    axes[1].plot(
        x,
        forgotten_mean,
        color=colors["forgetting"],
        marker="o",
        linewidth=2.0,
        markersize=3.5,
        zorder=3,
    )

    axes[1].set_ylabel(
        "Forgotten rate",
        fontsize=11,
    )

    # ==================================================
    # Soft forgetting
    # ==================================================
    axes[2].fill_between(
        x,
        margin_mean - margin_std,
        margin_mean + margin_std,
        color=colors["margin"],
        alpha=0.18,
        linewidth=0,
        zorder=1,
        )

    axes[2].plot(
        x,
        margin_mean,
        color=colors["margin"],
        marker="o",
        linewidth=2.0,
        markersize=3.5,
        zorder=3,
    )

    axes[2].set_ylabel(
        "Margin drop",
        fontsize=11,
    )

    # ==================================================
    # Epochs per task (automatic)
    # ==================================================
    epochs_per_task = int(g["curr_epoch"].max()) + 1

    task_boundaries = np.arange(
        epochs_per_task,
        int(x.max()) + 1,
        epochs_per_task,
        )

    if epochs_per_task <= 5:
        tick_step = 1          # MNIST/FashionMNIST
    elif epochs_per_task >= 40:
        tick_step = 10         # TinyImageNet (40 epochs/task)
    else:
        tick_step = 5          # CIFAR10 (15 epochs/task)

    handles = [
        Line2D(
            [0], [0],
            color=colors["flux"],
            marker="o",
            linestyle="None",
            markersize=6,
            label="Representation flux",
        ),
        Line2D(
            [0], [0],
            color=colors["forgetting"],
            marker="s",
            linestyle="None",
            markersize=6,
            label="Hard forgetting",
        ),
        Line2D(
            [0], [0],
            color=colors["margin"],
            marker="^",
            linestyle="None",
            markersize=6,
            label="Soft forgetting",
        ),
        Line2D(
            [0], [0],
            color="0.35",
            linestyle="--",
            linewidth=1.6,
            label="New task",
        ),
    ]

    axes[0].legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor="0.8",
        fancybox=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    for ax in axes:

        for boundary in task_boundaries:
            ax.axvline(
                boundary,
                color="0.45",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                zorder=0,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        ax.tick_params(
            axis="both",
            labelsize=10,
            width=0.8,
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.set_axisbelow(True)
        ax.margins(x=0)

    axes[-1].set_xlabel(
        "Global epoch",
        fontsize=11,
    )

    axes[-1].set_xticks(
        np.arange(
            0,
            int(x.max()) + 1,
            tick_step,
            )
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.92],
        h_pad=0.6,
        pad=0.25,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = output_dir / f"{DATASET}_probe_{probe}_multiseed_mean_std.png"
    pdf = output_dir / f"{DATASET}_probe_{probe}_multiseed_mean_std.pdf"

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    print(f"saved: {png}")
    print(f"saved: {pdf}")

def compute_smoothed_curve(
        df,
        target,
        feature="flux",
        grid_size=201,
        window_fraction=0.10,
        min_window=500,
):
    """
    Compute one smoothed target-vs-feature curve for one seed.

    Returns
    -------
    grid : np.ndarray
        Common normalized feature grid from 0 to 100.
    curve : np.ndarray
        Smoothed target values interpolated onto the common grid.
    """
    d = (
        df[[feature, target]]
        .dropna()
        .sort_values(feature)
        .reset_index(drop=True)
    )

    if len(d) < 100:
        return None, None

    # Use seed-specific 1st and 99th percentiles
    q01 = d[feature].quantile(0.01)
    q99 = d[feature].quantile(0.99)

    denom = q99 - q01

    if not np.isfinite(denom) or denom < 1e-12:
        return None, None

    d = d[
        (d[feature] >= q01)
        & (d[feature] <= q99)
        ].copy()

    if len(d) < 100:
        return None, None

    # Normalize each seed's feature range to 0–100
    d["feature_norm"] = (
            100.0
            * (d[feature] - q01)
            / denom
    )

    # A proportional window is better across differently sized seeds
    window = max(
        min_window,
        int(len(d) * window_fraction),
    )

    # Do not allow a window larger than the available data
    window = min(window, len(d))

    min_periods = max(
        50,
        window // 2,
        )

    smooth = (
        d[target]
        .rolling(
            window=window,
            center=True,
            min_periods=min_periods,
        )
        .mean()
    )

    x = d["feature_norm"].to_numpy(dtype=float)
    y = smooth.to_numpy(dtype=float)

    valid = (
            np.isfinite(x)
            & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return None, None

    # np.interp expects increasing, preferably unique x values
    unique_x, unique_indices = np.unique(
        x,
        return_index=True,
    )

    unique_y = y[unique_indices]

    if len(unique_x) < 2:
        return None, None

    grid = np.linspace(
        0.0,
        100.0,
        grid_size,
    )

    curve = np.interp(
        grid,
        unique_x,
        unique_y,
        left=np.nan,
        right=np.nan,
    )

    return grid, curve
def plot_smoothed_effect_across_seeds(
        seed_datasets,
        target,
        feature="flux",
        output_dir=None,
        filename=None,
        xlabel=None,
        ylabel=None,
        grid_size=201,
        window_fraction=0.10,
        min_window=500,
):
    """
    Compute one rolling curve per seed and then plot
    mean ± sample standard deviation across seeds.
    """
    if output_dir is None:
        raise ValueError("output_dir must be provided")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_map = {
        "forgotten": "Forgotten probability",
        "learned": "Learned probability",
        "conf_loss": "Confidence loss",
        "conf_gain": "Confidence gain",
        "margin_drop": "Margin drop",
    }

    if ylabel is None:
        ylabel = label_map.get(
            target,
            target.replace("_", " ").title(),
        )

    if xlabel is None:
        xlabel = (
            f"{feature.replace('_', ' ').title()} percentile"
        )

    curves = []
    used_seeds = []

    common_grid = None

    for seed_name, df_seed in seed_datasets.items():
        grid, curve = compute_smoothed_curve(
            df=df_seed,
            target=target,
            feature=feature,
            grid_size=grid_size,
            window_fraction=window_fraction,
            min_window=min_window,
        )

        if grid is None or curve is None:
            print(
                f"Skipped {seed_name} for "
                f"{target} vs {feature}"
            )
            continue

        common_grid = grid
        curves.append(curve)
        used_seeds.append(seed_name)

    if not curves:
        print(
            f"No valid seed curves for "
            f"{target} vs {feature}"
        )
        return

    curves = np.vstack(curves)

    # Some edge positions may be NaN for some seeds
    mean_curve = np.nanmean(
        curves,
        axis=0,
    )

    if curves.shape[0] > 1:
        std_curve = np.nanstd(
            curves,
            axis=0,
            ddof=1,
        )
    else:
        std_curve = np.zeros_like(
            mean_curve
        )

    n_at_x = np.sum(
        np.isfinite(curves),
        axis=0,
    )

    valid = (
            np.isfinite(common_grid)
            & np.isfinite(mean_curve)
            & np.isfinite(std_curve)
            & (n_at_x > 0)
    )

    x = common_grid[valid]
    mean_curve = mean_curve[valid]
    std_curve = std_curve[valid]
    n_at_x = n_at_x[valid]

    lower = mean_curve - std_curve
    upper = mean_curve + std_curve

    # Probability targets should stay within [0, 1]
    if target in {"forgotten", "learned"}:
        lower = np.clip(
            lower,
            0.0,
            1.0,
        )

        upper = np.clip(
            upper,
            0.0,
            1.0,
        )

    plt.figure(
        figsize=(3.4, 2.4)
    )

    plt.fill_between(
        x,
        lower,
        upper,
        alpha=0.18,
        linewidth=0,
        label="±1 std across seeds",
    )

    plt.plot(
        x,
        mean_curve,
        linewidth=1.5,
        label="Mean across seeds",
    )

    ax = plt.gca()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(0, 100)

    plt.xlabel(
        xlabel,
        fontsize=9,
    )

    plt.ylabel(
        ylabel,
        fontsize=9,
    )

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.legend(
        fontsize=6,
        frameon=False,
    )

    ax.grid(
        True,
        alpha=0.3,
    )
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.15)

    if filename is None:
        filename = (
            f"{target}_vs_{feature}"
            "_multiseed_mean_std.png"
        )

    png = output_dir / filename
    pdf = output_dir / filename.replace(
        ".png",
        ".pdf",
    )

    plt.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close()

    print(
        f"saved: {png} "
        f"using {len(used_seeds)} seeds"
    )

    print(f"saved: {pdf}")

def evaluate_auc(df, features):

    d = df[
        features + ["forgotten"]
        ].dropna()

    X = d[features].values
    y = d["forgotten"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(X_train, y_train)

    p = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, p)

    return auc

def build_dtm(Z0, Z1, K=16):

    Z = np.vstack([Z0, Z1])

    km = KMeans(
        n_clusters=K,
        random_state=0,
        n_init=10,
    )

    km.fit(Z)

    c0 = km.predict(Z0)
    c1 = km.predict(Z1)

    M = np.zeros((K, K))

    for a, b in zip(c0, c1):
        M[a, b] += 1

    row_sum = M.sum(axis=1, keepdims=True)

    M = np.divide(
        M,
        np.maximum(row_sum, 1),
    )

    return M


def dtm_stats(M):

    K = M.shape[0]

    stability = np.trace(M) / K

    leakage = 1.0 - stability

    eps = 1e-12

    entropy = -(
            M * np.log(M + eps)
    ).sum() / K

    return (
        stability,
        leakage,
        entropy,
    )

def compute_density(Z, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(Z)

    distances, _ = nn.kneighbors(Z)

    distances = distances[:, 1:]

    rho = k / (
            distances.sum(axis=1)
            + 1e-12
    )

    return rho

def parse_snapshot(path):
    name = path.name

    if name == "snapshot_init":
        return -1, -1, -1

    parts = name.split("_")

    if len(parts) >= 3 and parts[2].startswith("epoch"):
        task = int(parts[1].replace("task", ""))
        epoch = int(parts[2].replace("epoch", ""))

        # Determine epochs per task automatically from the snapshots
        root = path.parent

        epochs_this_task = []

        for p in root.glob(f"snapshot_task{task}_epoch*"):
            try:
                e = int(p.name.split("_")[2].replace("epoch", ""))
                epochs_this_task.append(e)
            except Exception:
                pass

        if epochs_this_task:
            epochs_per_task = max(epochs_this_task) + 1
        else:
            # Fallback
            epochs_per_task = 5

        global_epoch = task * epochs_per_task + epoch

        return task, epoch, global_epoch

    raise ValueError(f"Unknown snapshot name: {name}")


def snap_key(path):
    return parse_snapshot(path)[2]


def get_snapshots(root):
    return sorted(
        [p for p in root.glob("snapshot_*") if p.is_dir()],
        key=snap_key,
    )


def get_probes(snapshots):
    probe_ids = set()

    for snapshot in snapshots:
        for probe_dir in snapshot.glob("probe_task_*"):
            probe_ids.add(int(probe_dir.name.split("_")[-1]))

    return sorted(probe_ids)


def load_arr(snapshot, probe, name):
    path = snapshot / f"probe_task_{probe}" / name
    return np.load(path) if path.exists() else None


def transition_rows(prev_snap, curr_snap, probe):
    Z0 = load_arr(prev_snap, probe, f"{LAYER}.npy")
    Z1 = load_arr(curr_snap, probe, f"{LAYER}.npy")

    c0 = load_arr(prev_snap, probe, "correct.npy")
    c1 = load_arr(curr_snap, probe, "correct.npy")

    p0 = load_arr(prev_snap, probe, "prob_true.npy")
    p1 = load_arr(curr_snap, probe, "prob_true.npy")

    m0 = load_arr(prev_snap, probe, "margin.npy")
    m1 = load_arr(curr_snap, probe, "margin.npy")

    y = load_arr(curr_snap, probe, "labels.npy")

    required = [Z0, Z1, c0, c1, p0, p1, m0, m1, y]

    if any(x is None for x in required):
        return None

    if Z0.shape != Z1.shape:
        raise ValueError(
            f"Shape mismatch: {prev_snap.name} -> {curr_snap.name}: "
            f"{Z0.shape} vs {Z1.shape}"
        )

    prev_task, prev_epoch, prev_global = parse_snapshot(prev_snap)
    curr_task, curr_epoch, curr_global = parse_snapshot(curr_snap)

    flux = np.linalg.norm(Z1 - Z0, axis=1)
    M = build_dtm(Z0, Z1)

    stability, leakage, entropy = dtm_stats(M)
    rho0 = compute_density(Z0)
    rho1 = compute_density(Z1)

    delta_rho = rho1 - rho0
    forgotten = (c0 == 1) & (c1 == 0)
    learned = (c0 == 0) & (c1 == 1)
    stable_correct = (c0 == 1) & (c1 == 1)
    stable_wrong = (c0 == 0) & (c1 == 0)

    prob_true_drop = p0 - p1

    conf_loss = np.maximum(prob_true_drop, 0.0)
    conf_gain = np.maximum(-prob_true_drop, 0.0)

    return pd.DataFrame({
        "probe": probe,
        "sample": np.arange(len(flux)),
        "label": y,
        "snapshot_prev": prev_snap.name,
        "snapshot_curr": curr_snap.name,
        "prev_task": prev_task,
        "curr_task": curr_task,
        "prev_epoch": prev_epoch,
        "curr_epoch": curr_epoch,
        "global_epoch": curr_global,
        "transition": curr_global,
        "flux": flux,
        "correct_prev": c0,
        "correct_curr": c1,
        "forgotten": forgotten.astype(int),
        "learned": learned.astype(int),
        "stable_correct": stable_correct.astype(int),
        "stable_wrong": stable_wrong.astype(int),
        "prob_true_prev": p0,
        "prob_true_curr": p1,
        "prob_true_drop": prob_true_drop,
        "conf_loss": conf_loss,
        "conf_gain": conf_gain,
        "margin_prev": m0,
        "margin_curr": m1,
        "margin_drop": m0 - m1,
        "rho_prev": rho0,
        "rho_curr": rho1,
        "delta_rho": delta_rho,
        "stability": stability,
        "leakage": leakage,
        "entropy": entropy,
    })



def lagged_correlations(summary, max_lag=4):
    rows = []

    for probe, g in summary.groupby("probe"):
        g = g.sort_values("global_epoch").reset_index(drop=True)

        for lag in range(max_lag + 1):
            x = g["flux_mean"]

            y_forget = g["forgotten_rate"].shift(-lag)
            y_margin = g["margin_drop"].shift(-lag)
            y_prob = g["prob_true_drop"].shift(-lag)

            rows.append({
                "probe": probe,
                "lag": lag,
                "corr_flux_forgotten":
                    x.corr(y_forget),
                "corr_flux_margin_drop":
                    x.corr(y_margin),
                "corr_flux_prob_true_drop":
                    x.corr(y_prob),
                "n":
                    int((x.notna() & y_forget.notna()).sum()),
            })

    return pd.DataFrame(rows)


def plot_lagged_correlations(lagged):
    if lagged.empty:
        return

    mean_lagged = (
        lagged
        .groupby("lag")
        .agg(
            flux_forgotten=(
                "corr_flux_forgotten",
                "mean"
            ),
            flux_margin_drop=(
                "corr_flux_margin_drop",
                "mean"
            ),
            flux_prob_true_drop=(
                "corr_flux_prob_true_drop",
                "mean"
            ),
        )
        .reset_index()
    )

    plt.figure(figsize=(6.5, 4.0))

    plt.plot(
        mean_lagged["lag"],
        mean_lagged["flux_forgotten"],
        marker="o",
        linewidth=2.0,
        label="Forgotten rate",
    )

    plt.plot(
        mean_lagged["lag"],
        mean_lagged["flux_margin_drop"],
        marker="s",
        linewidth=2.0,
        label="Margin drop",
    )

    plt.plot(
        mean_lagged["lag"],
        mean_lagged["flux_prob_true_drop"],
        marker="^",
        linewidth=2.0,
        label="True-probability drop",
    )

    plt.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        alpha=0.4,
    )

    plt.xlabel("Lag")
    plt.ylabel("Correlation with flux(t)")
    plt.title("Lagged prediction from representation flux")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    png = OUT / f"{DATASET}_lagged_correlations.png"
    pdf = OUT / f"{DATASET}_lagged_correlations.pdf"

    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()

    print(f"saved: {png}")
    print(f"saved: {pdf}")
def build_dataset(root):
    snapshots = get_snapshots(root)
    probes = get_probes(snapshots)

    print(f"snapshots: {len(snapshots)}")
    print(f"probes: {probes}")

    rows = []

    for probe in probes:
        for prev_snap, curr_snap in zip(snapshots[:-1], snapshots[1:]):
            df = transition_rows(prev_snap, curr_snap, probe)

            if df is not None:
                rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)









def plot_smoothed_effect(
        df,
        target,
        feature="flux",
        filename=None,
        xlabel=None,
        ylabel=None,
        color="tab:red",
):
    d = (
        df[[feature, target]]
        .dropna()
        .sort_values(feature)
        .reset_index(drop=True)
    )

    if len(d) < 5000:
        return

    # --------------------------------------------------
    # Pretty labels
    # --------------------------------------------------
    label_map = {
        "forgotten": "Forgotten",
        "learned": "Learned",
        "conf_loss": "Confidence Loss",
        "conf_gain": "Confidence Gain",
    }

    if ylabel is None:
        ylabel = label_map.get(
            target,
            target.replace("_", " ").title()
        )

    if xlabel is None:
        xlabel = feature.replace("_", " ").title()

    # --------------------------------------------------
    # Remove extreme tails
    # --------------------------------------------------
    q01 = np.quantile(d[feature], 0.01)
    q99 = np.quantile(d[feature], 0.99)

    d = d.loc[
        (d[feature] >= q01)
        & (d[feature] <= q99)
        ].copy()

    # --------------------------------------------------
    # Normalize to [0,100]
    # --------------------------------------------------
    denom = q99 - q01

    if denom < 1e-12:
        return

    d["feature_norm"] = (
            100.0
            * (d[feature] - q01)
            / denom
    )

    # --------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------
    window = 5000

    rolling = d[target].rolling(
        window,
        center=True,
        min_periods=window // 2,
    )

    mean = rolling.mean()
    std = rolling.std()
    count = rolling.count()

    # --------------------------------------------------
    # 95% confidence interval
    # --------------------------------------------------
    se = std / np.sqrt(
        np.maximum(count, 1)
    )

    lower = mean - 1.96 * se
    upper = mean + 1.96 * se

    x = d["feature_norm"].to_numpy()

    valid = (
            np.isfinite(x)
            & np.isfinite(mean)
            & np.isfinite(lower)
            & np.isfinite(upper)
    )

    x = x[valid]
    mean = mean[valid]
    lower = lower[valid]
    upper = upper[valid]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(3.4, 2.4))

    plt.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )

    plt.plot(
        x,
        mean,
        color=color,
        linewidth=1.0,
        zorder=3,
    )

    ax = plt.gca()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    plt.xlabel(xlabel, fontsize=9)
    plt.ylabel(ylabel, fontsize=9)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    ax.set_xlim(0, 100)
    ax.margins(x=0, y=0)

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------
    line_handle = Line2D(
        [0],
        [0],
        color=color,
        lw=1.0,
        label=f"{ylabel} (w={window})",
    )

    band_handle = Patch(
        facecolor=color,
        alpha=0.10,
        edgecolor="none",
        label=f"95% CI (w={window})",
    )

    legend = plt.legend(
        handles=[line_handle, band_handle],
        loc="upper left",
        fontsize=6,
        frameon=True,
        framealpha=0.80,
        borderpad=0.15,
        handlelength=1.4,
        labelspacing=0.2,
    )

    legend.get_frame().set_linewidth(0.5)

    plt.grid(False)
    plt.tight_layout(pad=0.1)

    if filename is None:
        filename = f"{target}_vs_{feature}.png"

    png = OUT / filename
    pdf = OUT / filename.replace(".png", ".pdf")

    plt.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close()

    print(f"saved: {png}")
    print(f"saved: {pdf}")

def summarize(df):
    return (
        df.groupby(
            ["probe", "global_epoch", "curr_task", "curr_epoch"]
        )
        .agg(
            flux_mean=("flux", "mean"),
            flux_std=("flux", "std"),

            forgotten_rate=("forgotten", "mean"),
            forgotten_std=("forgotten", "std"),

            learned_rate=("learned", "mean"),
            learned_std=("learned", "std"),

            prob_true_drop=("prob_true_drop", "mean"),

            margin_drop=("margin_drop", "mean"),
            margin_std=("margin_drop", "std"),

            n=("sample", "count"),
        )
        .reset_index()
    )


def plot_probe_summary(summary, probe):

    g = summary[summary["probe"] == probe].copy()

    if g.empty:
        return

    g = g.sort_values("global_epoch")

    x = g["global_epoch"].to_numpy()

    flux = g["flux_mean"].to_numpy()
    flux_std = g["flux_std"].fillna(0).to_numpy()

    forgetting = g["forgotten_rate"].to_numpy()

    margin = g["margin_drop"].to_numpy()
    margin_std = g["margin_std"].fillna(0).to_numpy()

    n = g["n"].to_numpy()

    # ---------------------------------------
    # 95% CI for forgetting rate
    # ---------------------------------------
    se_forgetting = np.sqrt(
        np.maximum(
            forgetting * (1.0 - forgetting) / np.maximum(n, 1),
            0.0,
            )
    )

    forgetting_ci = 1.96 * se_forgetting

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 6.8),
        sharex=True,
    )

    # =======================================
    # Flux
    # =======================================
    ax = axes[0]

    ax.fill_between(
        x,
        flux - flux_std,
        flux + flux_std,
        color="tab:blue",
        alpha=0.20,
        linewidth=0,
        )

    ax.plot(
        x,
        flux,
        color="tab:blue",
        marker="o",
        linewidth=2.0,
        markersize=5,
    )

    ax.set_ylabel("Mean flux")
    ax.set_title("Representation flux")

    # =======================================
    # Hard forgetting
    # =======================================
    ax = axes[1]

    ax.fill_between(
        x,
        np.maximum(forgetting - forgetting_ci, 0.0),
        np.minimum(forgetting + forgetting_ci, 1.0),
        color="tab:red",
        alpha=0.20,
        linewidth=0,
    )

    ax.plot(
        x,
        forgetting,
        color="tab:red",
        marker="s",
        linewidth=2.0,
        markersize=5,
    )

    ax.set_ylabel("Forgotten rate")
    ax.set_title("Hard forgetting")

    # =======================================
    # Soft forgetting
    # =======================================
    ax = axes[2]

    ax.fill_between(
        x,
        margin - margin_std,
        margin + margin_std,
        color="tab:green",
        alpha=0.20,
        linewidth=0,
        )

    ax.plot(
        x,
        margin,
        color="tab:green",
        marker="^",
        linewidth=2.0,
        markersize=5,
    )

    ax.set_ylabel("Margin drop")
    ax.set_title("Soft forgetting")

    # =======================================
    # Task boundaries
    # =======================================
    task_boundaries = []

    for task in sorted(g["curr_task"].unique()):
        first = g[g["curr_task"] == task]["global_epoch"].min()
        task_boundaries.append((task, first))

    for ax in axes:

        for task, first in task_boundaries:
            ax.axvline(
                first,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.35,
            )

        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Global epoch")

    max_epoch = int(g["global_epoch"].max())

    axes[-1].set_xticks(
        np.arange(0, max_epoch + 1, 1)
    )

    fig.suptitle(
        f"Probe {probe}: flux and forgetting dynamics",
        fontsize=14,
        y=0.995,
    )

    fig.tight_layout()

    png = OUT / f"{DATASET}_probe_{probe}_flux_forgetting_clean.png"
    pdf = OUT / f"{DATASET}_probe_{probe}_flux_forgetting_clean.pdf"

    plt.savefig(png, dpi=300)
    plt.savefig(pdf)

    plt.close()

    print(f"saved: {png}")
    print(f"saved: {pdf}")

def plot_2d_heatmap(
        df,
        feature_x,
        feature_y,
        target,
        filename,
        title=None,
        bins=20,
):

    d = df[
        [feature_x, feature_y, target]
    ].dropna().copy()

    # Convert both features to percentiles (0-100)
    d["x_percentile"] = d[feature_x].rank(pct=True) * 100.0
    d["y_percentile"] = d[feature_y].rank(pct=True) * 100.0

    d["x_bin"] = pd.cut(
        d["x_percentile"],
        bins=np.linspace(0, 100, bins + 1),
        labels=False,
        include_lowest=True,
    )

    d["y_bin"] = pd.cut(
        d["y_percentile"],
        bins=np.linspace(0, 100, bins + 1),
        labels=False,
        include_lowest=True,
    )

    heat = (
        d.groupby(
            ["y_bin", "x_bin"]
        )[target]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(7.2, 6.1))

    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    rocket = sns.color_palette("rocket", as_cmap=True)

    x = np.linspace(0, 1, 256)
    x_new = x ** 0.55

    rocket_stretched = LinearSegmentedColormap.from_list(
        "rocket_stretched",
        rocket(x_new)
    )

    im = plt.imshow(
        heat,
        origin="lower",
        aspect="auto",
        cmap=rocket_stretched,
    )

    plt.colorbar(
        im,
        label=f"P({target})"
    )

    label_map = {
        "flux": "Representation flux percentile",
        "rho_prev": "Representation density percentile",
        "delta_rho": "Density change percentile",
        "stability": "Stability percentile",
        "leakage": "Leakage percentile",
        "entropy": "Transition entropy percentile",
    }

    plt.xlabel(label_map.get(feature_x, feature_x), fontsize=11)
    plt.ylabel(label_map.get(feature_y, feature_y), fontsize=11)

    # Percentile ticks (0–100)
    tick_positions = np.linspace(0, bins - 1, 6)
    tick_labels = ["0", "20", "40", "60", "80", "100"]

    plt.xticks(
        tick_positions,
        tick_labels,
        fontsize=10,
    )

    plt.yticks(
        tick_positions,
        tick_labels,
        fontsize=10,
    )

    plt.tight_layout()

    png = OUT / filename
    pdf = OUT / filename.replace(
        ".png",
        ".pdf",
    )

    plt.savefig(png, dpi=300)
    plt.savefig(pdf)

    plt.close()

    print(f"saved: {png}")
    print(f"saved: {pdf}")

def plot_flux_vs_forgetting(df):
    summary = (
        df.groupby(["probe", "snapshot_prev", "snapshot_curr"])
        .agg(
            flux_mean=("flux", "mean"),
            forgotten_rate=("forgotten", "mean"),
            margin_drop=("margin_drop", "mean"),
            prob_true_drop=("prob_true_drop", "mean"),
        )
        .reset_index()
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(summary["flux_mean"], summary["forgotten_rate"])

    plt.xlabel("Mean flux")
    plt.ylabel("Forgotten rate")
    plt.title("Epoch-level flux vs forgetting")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    png = OUT / f"{DATASET}_flux_vs_forgotten_rate.png"
    pdf = OUT / f"{DATASET}_flux_vs_forgotten_rate.pdf"

    plt.savefig(png, dpi=300)
    plt.savefig(pdf)


    print(f"saved: {png}")
    print(f"saved: {pdf}")





def main():
    df = build_dataset(ROOT)

    if df.empty:
        print("No data found. Did you save correct/prob_true/margin arrays?")
        return

    sample_csv = OUT / f"{DATASET}_sample_forgetting_flux.csv"
    df.to_csv(sample_csv, index=False)

    summary = summarize(df)

    summary_csv = OUT / f"{DATASET}_transition_forgetting_flux.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"saved: {sample_csv}")
    print(f"saved: {summary_csv}")
    lagged = lagged_correlations(
        summary,
        max_lag=4
    )

    lagged_csv = OUT / f"{DATASET}_lagged_correlations.csv"
    lagged.to_csv(lagged_csv, index=False)

    plot_lagged_correlations(lagged)

    print()
    print("Lagged correlations")
    print(lagged)
    print(f"saved: {lagged_csv}")

    for probe in sorted(df["probe"].unique()):
        plot_probe_summary(summary, probe)

    plot_flux_vs_forgetting(df)

    plot_smoothed_effect(
        df,
        target="forgotten",
        feature="stability",
        filename=f"{DATASET}_forgotten_vs_stability.png",
    )

    plot_smoothed_effect(
        df,
        target="forgotten",
        feature="leakage",
        filename=f"{DATASET}_forgotten_vs_leakage.png",
    )

    plot_smoothed_effect(
        df,
        target="conf_loss",
        feature="stability",
        filename=f"{DATASET}_conf_loss_vs_stability.png",
    )

    plot_smoothed_effect(
        df,
        target="forgotten",
        feature="entropy",
        filename=f"{DATASET}_forgotten_vs_entropy.png",
    )

    plot_smoothed_effect(
        df,
        target="learned",
        filename=f"{DATASET}_learned_vs_flux_smooth.png",
    )

    plot_smoothed_effect(
        df,
        target="forgotten",
        feature="rho_prev",
        filename=f"{DATASET}_forgotten_vs_density.png",
        xlabel="Normalized Density",
    )

    plot_smoothed_effect(
        df,
        target="forgotten",
        feature="delta_rho",
        filename=f"{DATASET}_forgotten_vs_delta_density.png",
        xlabel="Density Change",
    )

    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="leakage",
        target="forgotten",
        filename=f"{DATASET}_forgotten_flux_leakage_heatmap.png",
        title="P(forgotten | flux, leakage)",
    )

    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="stability",
        target="forgotten",
        filename=f"{DATASET}_forgotten_flux_stability_heatmap.png",
        title="P(forgotten | flux, stability)",
    )

    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="entropy",
        target="forgotten",
        filename=f"{DATASET}_forgotten_flux_entropy_heatmap.png",
        title="P(forgotten | flux, entropy)",
    )

    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="leakage",
        target="conf_loss",
        filename=f"{DATASET}_conf_loss_flux_leakage_heatmap.png",
        title="Confidence loss | flux, leakage",
    )

    plot_smoothed_effect(
        df,
        target="conf_loss",
        feature="rho_prev",
        filename=f"{DATASET}_conf_loss_vs_density.png",
        xlabel="Normalized Density",
    )

    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="rho_prev",
        target="forgotten",
        filename=f"{DATASET}_forgotten_flux_density_heatmap.png",
        title="P(forgotten | flux, density)",
    )

    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="rho_prev",
        target="learned",
        filename=f"{DATASET}_learned_flux_density_heatmap.png",
        title="P(learned | flux, density)",
    )

    corr = df[
        [
            "flux",
            "rho_prev",
            "delta_rho",
            "stability",
            "leakage",
            "entropy",
            "forgotten",
            "conf_loss",
            "conf_gain",
            "margin_drop",
        ]
    ].corr()

    corr_csv = OUT / f"{DATASET}_correlations.csv"
    corr.to_csv(corr_csv)

    print()
    print(corr)
    print(f"saved: {corr_csv}")
    print(f"done: {OUT}")


    models = {
        "Flux":
            ["flux"],

        "Density":
            ["rho_prev"],

        "Flux+Density":
            ["flux", "rho_prev"],

        "Flux+Density+Leakage":
            ["flux", "rho_prev", "leakage"],

        "Flux+Density+Stability":
            ["flux", "rho_prev", "stability"],

        "Flux+Density+Entropy":
            ["flux", "rho_prev", "entropy"],

        "All":
            [
                "flux",
                "rho_prev",
                "delta_rho",
                "stability",
                "leakage",
                "entropy",
            ],
    }
    root_parent = ROOT.parent

    seed_dirs = sorted(root_parent.glob("replay_seed*"))

    multiseed_summary, seed_datasets = (
        build_multiseed_summary(seed_dirs)
    )

    aggregated = aggregate_summary_across_seeds(multiseed_summary)

    multiseed_out = root_parent / "forgetting_flux_multiseed"
    multiseed_out.mkdir(exist_ok=True)

    for probe in sorted(aggregated["probe"].unique()):
        plot_probe_summary_across_seeds(
            aggregated,
            probe,
            multiseed_out,
        )

    # ============================================================
    # Multiseed smoothed curves: mean ± std across seeds
    # ============================================================

    plot_smoothed_effect_across_seeds(
        seed_datasets,
        target="forgotten",
        feature="flux",
        output_dir=multiseed_out,
        filename=f"{DATASET}_forgotten_vs_flux_multiseed_mean_std.png",
        xlabel="Flux percentile",
        ylabel="Forgetting probability",
    )

    plot_smoothed_effect_across_seeds(
        seed_datasets,
        target="learned",
        feature="flux",
        output_dir=multiseed_out,
        filename=f"{DATASET}_learned_vs_flux_multiseed_mean_std.png",
        xlabel="Flux percentile",
        ylabel="Learning probability",
    )

    plot_smoothed_effect_across_seeds(
        seed_datasets,
        target="conf_loss",
        feature="flux",
        output_dir=multiseed_out,
        filename=f"{DATASET}_conf_loss_vs_flux_multiseed_mean_std.png",
        xlabel="Flux percentile",
        ylabel="Confidence loss",
    )

    plot_smoothed_effect_across_seeds(
        seed_datasets,
        target="conf_gain",
        feature="flux",
        output_dir=multiseed_out,
        filename=f"{DATASET}_conf_gain_vs_flux_multiseed_mean_std.png",
        xlabel="Flux percentile",
        ylabel="Confidence gain",
    )

    plot_smoothed_effect_across_seeds(
        seed_datasets,
        target="forgotten",
        feature="rho_prev",
        output_dir=multiseed_out,
        filename=f"{DATASET}_forgotten_vs_density_multiseed_mean_std.png",
        xlabel="Density percentile",
        ylabel="Forgetting probability",
    )

    plot_smoothed_effect_across_seeds(
        seed_datasets,
        target="forgotten",
        feature="delta_rho",
        output_dir=multiseed_out,
        filename=f"{DATASET}_forgotten_vs_delta_density_multiseed_mean_std.png",
        xlabel="Density-change percentile",
        ylabel="Forgetting probability",
    )

    plot_smoothed_effect_across_seeds(
        seed_datasets,
        target="forgotten",
        feature="stability",
        output_dir=multiseed_out,
        filename=f"{DATASET}_forgotten_vs_stability_multiseed_mean_std.png",
        xlabel="Stability percentile",
        ylabel="Forgetting probability",
    )

    plot_smoothed_effect_across_seeds(
        seed_datasets,
        target="forgotten",
        feature="leakage",
        output_dir=multiseed_out,
        filename=f"{DATASET}_forgotten_vs_leakage_multiseed_mean_std.png",
        xlabel="Leakage percentile",
        ylabel="Forgetting probability",
    )

    plot_smoothed_effect_across_seeds(
        seed_datasets,
        target="forgotten",
        feature="entropy",
        output_dir=multiseed_out,
        filename=f"{DATASET}_forgotten_vs_entropy_multiseed_mean_std.png",
        xlabel="Transition entropy percentile",
        ylabel="Forgetting probability",
    )

    plot_smoothed_effect_across_seeds(
        seed_datasets,
        target="conf_loss",
        feature="rho_prev",
        output_dir=multiseed_out,
        filename=f"{DATASET}_conf_loss_vs_density_multiseed_mean_std.png",
        xlabel="Density percentile",
        ylabel="Confidence loss",
    )

    plot_smoothed_effect_across_seeds(
        seed_datasets,
        target="conf_loss",
        feature="stability",
        output_dir=multiseed_out,
        filename=f"{DATASET}_conf_loss_vs_stability_multiseed_mean_std.png",
        xlabel="Stability percentile",
        ylabel="Confidence loss",
    )
    root_parent = ROOT.parent

    seed_dirs = sorted(
        root_parent.glob("replay_seed*")
    )

    print()
    print("Found seeds:")
    for s in seed_dirs:
        print(" ", s.name)

    all_results = defaultdict(list)
    auc_seed_rows = []

    for seed_dir in seed_dirs:

        print(f"\nProcessing {seed_dir.name}")

        df = seed_datasets.get(seed_dir.name)

        if df is None or df.empty:
            print("  skipped (no data)")
            continue

        for name, features in models.items():

            auc = evaluate_auc(
                df,
                features,
            )

            all_results[name].append(auc)

            auc_seed_rows.append({
                "seed": seed_dir.name,
                "model": name,
                "auc": auc,
            })
    auc_by_seed = pd.DataFrame(
        auc_seed_rows
    )

    auc_by_seed_csv = (
            root_parent
            / f"{DATASET}_forgetting_prediction_auc_by_seed.csv"
    )

    auc_by_seed.to_csv(
        auc_by_seed_csv,
        index=False,
    )

    print()
    print("AUC BY SEED")
    print(auc_by_seed)
    print(f"saved: {auc_by_seed_csv}")
    rows = []

    for name, values in all_results.items():

        rows.append({
            "model": name,
            "auc_mean": np.mean(values),
            "auc_std": np.std(values, ddof=1),
            "n": len(values),
        })

    results = (
            pd.DataFrame(rows)
            .sort_values("auc_mean", ascending=False)
    )

    csv = root_parent / f"{DATASET}_forgetting_prediction_auc.csv"
    results.to_csv(csv, index=False)

    print()
    print("FORGETTING PREDICTION")
    print(results)

    print(f"saved: {csv}")

if __name__ == "__main__":
    main()