import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

ROOT = Path("data/mnist/mlp/replay")
OUT = ROOT / "dtm"
OUT.mkdir(parents=True, exist_ok=True)

LAYER = "layer3"
K = 16
def plot_density_evolution_task(
        snapshot_dirs,
        probe_id,
        task_id,
        layer=LAYER
):

    selected = []
    out_dir = (
            OUT /
            f"probe_{probe_id}"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    for snapshot in snapshot_dirs:

        name = snapshot.name

        if task_id == 0:
            if not name.startswith(
                    "snapshot_task0_"
            ):
                continue

        else:
            prefix = (
                f"snapshot_task"
                f"{task_id}_"
            )

            if not name.startswith(
                    prefix
            ):
                continue

        selected.append(
            snapshot
        )

    if len(selected) == 0:
        return

    plt.figure(figsize=(8, 6))

    all_rho = []
    labels = []

    for snapshot in selected:

        Z = load_representation(
            snapshot,
            probe_id,
            layer
        )

        if Z is None:
            print(
                f"Missing representation:"
                f" probe={probe_id}"
                f" snapshot={snapshot.name}"
            )
            continue

        rho = compute_density(Z)
        csv_path = (
                out_dir /
                f"density_{snapshot.name}.csv"
        )

        np.savetxt(
            csv_path,
            rho,
            delimiter=",",
            header="density",
            comments=""
        )
        all_rho.append(rho)

        labels.append(
            snapshot.name.replace(
                f"snapshot_task{task_id}_",
                ""
            )
        )

    if len(all_rho) == 0:
        plt.close()
        return

    global_min = min(
        rho.min()
        for rho in all_rho
    )

    global_max = max(
        rho.max()
        for rho in all_rho
    )

    bins = np.linspace(
        global_min,
        global_max,
        50
    )

    for rho, label in zip(
            all_rho,
            labels
    ):

        plt.hist(
            rho,
            bins=bins,
            density=True,
            alpha=0.25,
            label=label
        )

    plt.xlabel("Density")
    plt.ylabel("Probability Density")

    plt.title(
        f"Probe {probe_id} "
        f"during Task {task_id}"
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()



    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        out_dir /
        f"density_task_{task_id}.png",
        dpi=300
    )

    plt.close()

def compute_density(Z, k=10):

    nn = NearestNeighbors(
        n_neighbors=k + 1
    )

    nn.fit(Z)

    distances, _ = nn.kneighbors(Z)

    distances = distances[:, 1:]

    rho = k / (
            distances.sum(axis=1)
            + 1e-12
    )

    return rho

def transport_stats(
        M,
        centers=None,
        assignments=None
):

    K = M.shape[0]

    stability = (
            np.trace(M)
            / K
    )

    leakage = 1.0 - stability

    entropy = -np.sum(
        M * np.log(M + 1e-12)
    ) / K

    stats = {
        "stability": float(stability),
        "leakage": float(leakage),
        "entropy": float(entropy)
    }

    if (
            centers is not None
            and assignments is not None
    ):

        p = np.bincount(
            assignments,
            minlength=K
        ).astype(np.float64)

        p = p / np.maximum(
            p.sum(),
            1e-12
        )

        D = np.linalg.norm(
            centers[:, None, :]
            - centers[None, :, :],
            axis=-1
        )

        F = (
                p[:, None]
                * M
        )

        offdiag_flux = (
                np.sum(F)
                - np.trace(F)
        )

        geometric_flux = np.sum(
            F * D
        )

        stats["flux"] = float(
            offdiag_flux
        )

        stats["geometric_flux"] = float(
            geometric_flux
        )

    return stats
def get_snapshot_index(name):

    if name == "snapshot_init":
        return -1

    parts = name.split("_")

    # old format: snapshot_0
    if len(parts) == 2:
        return int(parts[1])

    # new format: snapshot_task0_epoch0
    task = int(parts[1].replace("task", ""))
    epoch = int(parts[2].replace("epoch", ""))

    return task * 5 + epoch


def load_representation(
        snapshot_dir,
        probe_id,
        layer=LAYER
):

    path = (
            snapshot_dir
            / f"probe_task_{probe_id}"
            / f"{layer}.npy"
    )

    if not path.exists():
        return None

    return np.load(path)


def build_dtm(
        Z1,
        Z2,
        k=K
):

    Z = np.concatenate(
        [Z1, Z2],
        axis=0
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=0,
        n_init=10
    )

    labels = kmeans.fit_predict(Z)

    centers = kmeans.cluster_centers_

    c1 = labels[:len(Z1)]
    c2 = labels[len(Z1):]

    M = np.zeros(
        (k, k),
        dtype=np.float64
    )

    for i, j in zip(c1, c2):
        M[i, j] += 1

    row_sum = M.sum(
        axis=1,
        keepdims=True
    )

    M /= np.maximum(
        row_sum,
        1e-12
    )

    return M, centers, c1





def save_heatmap(
        M,
        title,
        path
):

    plt.figure(figsize=(6, 5))

    plt.imshow(
        M,
        aspect="auto"
    )

    plt.colorbar()

    plt.title(title)

    plt.xlabel(
        "Cluster t+1"
    )

    plt.ylabel(
        "Cluster t"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


def plot_curve(
        xs,
        ys,
        ylabel,
        filename
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        xs,
        ys,
        marker="o"
    )

    plt.xlabel("Transition")
    plt.ylabel(ylabel)

    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=300
    )

    plt.close()


def main():

    snapshot_dirs = sorted(
        ROOT.glob("snapshot_*"),
        key=lambda x:
        get_snapshot_index(x.name)
    )

    if len(snapshot_dirs) < 2:
        print(
            "Need at least two snapshots"
        )
        return

    max_probe = 0

    for s in snapshot_dirs:

        for p in s.glob(
                "probe_task_*"
        ):

            pid = int(
                p.name.split("_")[-1]
            )

            max_probe = max(
                max_probe,
                pid
            )

    for probe_id in range(
            max_probe + 1
    ):

        print(
            f"\nProcessing Task "
            f"{probe_id}"
        )

        stability_curve = []
        leakage_curve = []
        entropy_curve = []
        flux_curve = []
        transition_ids = []

        transition_idx = 0

        for i in range(
                len(snapshot_dirs) - 1
        ):

            s1 = snapshot_dirs[i]
            s2 = snapshot_dirs[i + 1]

            Z1 = load_representation(
                s1,
                probe_id
            )

            Z2 = load_representation(
                s2,
                probe_id
            )

            if (
                    Z1 is None
                    or
                    Z2 is None
            ):
                continue

            M, centers, assignments = build_dtm(
                Z1,
                Z2,
                k=K,
            )

            stats = transport_stats(
                M,
                centers=centers,
                assignments=assignments,
            )
            stability_curve.append(
                stats["stability"]
            )

            leakage_curve.append(
                stats["leakage"]
            )

            entropy_curve.append(
                stats["entropy"]
            )
            flux_curve.append(
                stats["flux"]
            )

            transition_ids.append(
                transition_idx
            )

            transition_idx += 1

            heatmap_dir = (
                    OUT
                    / f"task_{probe_id}"
            )

            heatmap_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            save_heatmap(
                M,
                (
                    f"Task {probe_id + 1}\n"
                    f"{s1.name}"
                    " -> "
                    f"{s2.name}"
                ),
                heatmap_dir
                /
                f"dtm_{transition_idx}.png"
            )

        if len(
                transition_ids
        ) == 0:
            continue

        plot_curve(
            transition_ids,
            stability_curve,
            "Stability",
            (
                f"task_"
                f"{probe_id}_"
                f"stability.png"
            )
        )

        plot_curve(
            transition_ids,
            leakage_curve,
            "Leakage",
            (
                f"task_"
                f"{probe_id}_"
                f"leakage.png"
            )
        )

        plot_curve(
            transition_ids,
            entropy_curve,
            "Entropy",
            (
                f"task_"
                f"{probe_id}_"
                f"entropy.png"
            )
        )
        plot_curve(
            transition_ids,
            flux_curve,
            "Flux",
            (
                f"task_"
                f"{probe_id}_"
                f"flux.png"
            )
        )
        for task_id in range(
                probe_id,
                max_probe + 1
        ):
            plot_density_evolution_task(
                snapshot_dirs,
                probe_id,
                task_id
            )

        summary = {

            "stability_mean":
                float(
                    np.mean(
                        stability_curve
                    )
                ),

            "leakage_mean":
                float(
                    np.mean(
                        leakage_curve
                    )
                ),

            "entropy_mean":
                float(
                    np.mean(
                        entropy_curve
                    )
                ),

            "flux_mean":
                float(
                    np.mean(
                        flux_curve
                    )
                )
        }

        with open(
                OUT
                /
                f"task_"
                f"{probe_id}_summary.json",
                "w"
        ) as f:

            json.dump(
                summary,
                f,
                indent=4
            )

    print()
    print(
        f"Results saved to:\n{OUT}"
    )


if __name__ == "__main__":
    main()