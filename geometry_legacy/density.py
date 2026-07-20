from sklearn.neighbors import NearestNeighbors
import numpy as np


class DensityAnalyzer:

    def compute(self, Z, k=10):

        nbrs = NearestNeighbors(
            n_neighbors=k
        ).fit(Z)

        dist, _ = nbrs.kneighbors(Z)

        density = 1.0 / (
                dist[:, 1:].mean(axis=1) + 1e-8
        )

        return {
            "mean": density.mean(),
            "std": density.std()
        }