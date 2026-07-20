from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
ROOT = Path("data/mnist/mlp/replay")
OUT = ROOT / "forgetting_flux"
LAYER = "layer3"

OUT.mkdir(parents=True, exist_ok=True)
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

        # EPOCHS = 5 in splitmnist_geometry_buffer.py
        global_epoch = task * 5 + epoch

        return task, epoch, global_epoch

    raise ValueError(f"Unknown snapshot name: {name}")


def snap_key(path):
    return parse_snapshot(path)[2]


def get_snapshots():
    return sorted(
        [p for p in ROOT.glob("snapshot_*") if p.is_dir()],
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

def plot_learning_forgetting(df):
    d = (
        df[["flux", "forgotten", "learned"]]
        .dropna()
        .sort_values("flux")
    )

    window = max(500, len(d) // 100)

    forgot = (
        d["forgotten"]
        .rolling(window, center=True)
        .mean()
    )

    learned = (
        d["learned"]
        .rolling(window, center=True)
        .mean()
    )

    plt.figure(figsize=(7,5))

    plt.plot(
        d["flux"],
        forgot,
        label="P(forgotten | flux)",
        linewidth=2,
    )

    plt.plot(
        d["flux"],
        learned,
        label="P(learned | flux)",
        linewidth=2,
    )

    plt.xlabel("Flux")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT / "learning_forgetting_vs_flux.png",
        dpi=300,
        )
    plt.close()

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

    png = OUT / "lagged_correlations.png"
    pdf = OUT / "lagged_correlations.pdf"

    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()

    print(f"saved: {png}")
    print(f"saved: {pdf}")
def build_dataset():
    snapshots = get_snapshots()
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





def plot_raw_effect(df, target, filename):
    d = df[["flux", target]].dropna()

    plt.figure(figsize=(6,5))
    plt.scatter(
        d["flux"],
        d[target],
        s=1,
        alpha=0.02
    )

    plt.xlabel("Flux")
    plt.ylabel(target)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUT / filename, dpi=300)
    plt.close()

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt


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

    png = OUT / f"probe_{probe}_flux_forgetting_clean.png"
    pdf = OUT / f"probe_{probe}_flux_forgetting_clean.pdf"

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

    d["x_bin"] = pd.qcut(
        d[feature_x],
        q=bins,
        labels=False,
        duplicates="drop",
    )

    d["y_bin"] = pd.qcut(
        d[feature_y],
        q=bins,
        labels=False,
        duplicates="drop",
    )

    heat = (
        d.groupby(
            ["y_bin", "x_bin"]
        )[target]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        heat,
        origin="lower",
        aspect="auto",
        cmap="magma",
    )

    plt.colorbar(
        im,
        label=f"P({target})"
    )

    plt.xlabel(
        f"{feature_x} percentile"
    )

    plt.ylabel(
        f"{feature_y} percentile"
    )

    if title is None:
        title = (
            f"{target} | "
            f"{feature_x}, {feature_y}"
        )

    plt.title(title)

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

    path = OUT / "flux_vs_forgotten_rate.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"saved: {path}")





def main():
    df = build_dataset()

    if df.empty:
        print("No data found. Did you save correct/prob_true/margin arrays?")
        return

    sample_csv = OUT / "sample_forgetting_flux.csv"
    df.to_csv(sample_csv, index=False)

    summary = summarize(df)

    summary_csv = OUT / "transition_forgetting_flux.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"saved: {sample_csv}")
    print(f"saved: {summary_csv}")
    lagged = lagged_correlations(
        summary,
        max_lag=4
    )

    lagged_csv = OUT / "lagged_correlations.csv"
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
        target="conf_gain",
        filename="conf_gain_vs_flux_smooth.png",
    )

    plot_smoothed_effect(
        df,
        target="conf_loss",
        filename="conf_loss_vs_flux_smooth.png",
    )

    plot_smoothed_effect(
        df,
        target="forgotten",
        filename="forgotten_vs_flux_smooth.png",
    )

    plot_smoothed_effect(
        df,
        target="forgotten",
        filename="forgotten_vs_flux_smooth.png",
    )

    plot_smoothed_effect(
        df,
        target="forgotten",
        feature="stability",
        filename="forgotten_vs_stability.png",
    )
    plot_smoothed_effect(
        df,
        target="forgotten",
        feature="leakage",
        filename="forgotten_vs_leakage.png",
    )
    plot_smoothed_effect(
        df,
        target="conf_loss",
        feature="stability",
        filename="conf_loss_vs_stability.png",
    )
    plot_smoothed_effect(
        df,
        target="forgotten",
        feature="entropy",
        filename="forgotten_vs_entropy.png",
    )
    plot_smoothed_effect(
        df,
        target="learned",
        filename="learned_vs_flux_smooth.png",
    )
    plot_smoothed_effect(
        df,
        target="forgotten",
        feature="rho_prev",
        filename="forgotten_vs_density.png",
        xlabel="Normalized Density",
    )

    plot_smoothed_effect(
        df,
        target="forgotten",
        feature="delta_rho",
        filename="forgotten_vs_delta_density.png",
        xlabel="Density Change",
    )
    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="leakage",
        target="forgotten",
        filename="forgotten_flux_leakage_heatmap.png",
        title="P(forgotten | flux, leakage)",
    )
    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="stability",
        target="forgotten",
        filename="forgotten_flux_stability_heatmap.png",
        title="P(forgotten | flux, stability)",
    )
    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="entropy",
        target="forgotten",
        filename="forgotten_flux_entropy_heatmap.png",
        title="P(forgotten | flux, entropy)",
    )
    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="leakage",
        target="conf_loss",
        filename="conf_loss_flux_leakage_heatmap.png",
        title="Confidence loss | flux, leakage",
    )
    plot_smoothed_effect(
        df,
        target="conf_loss",
        feature="rho_prev",
        filename="conf_loss_vs_density.png",
        xlabel="Normalized Density",
    )
    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="rho_prev",
        target="forgotten",
        filename="forgotten_flux_density_heatmap.png",
        title="P(forgotten | flux, density)",
    )

    plot_2d_heatmap(
        df,
        feature_x="flux",
        feature_y="rho_prev",
        target="learned",
        filename="learned_flux_density_heatmap.png",
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

    corr_csv = OUT / "correlations.csv"
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
    rows = []

    for name, features in models.items():

        auc = evaluate_auc(
            df,
            features,
        )

        rows.append({
            "model": name,
            "auc": auc,
        })

    results = (
        pd.DataFrame(rows)
        .sort_values("auc", ascending=False)
    )

    print()
    print("FORGETTING PREDICTION AUC")
    print(results)

if __name__ == "__main__":
    main()