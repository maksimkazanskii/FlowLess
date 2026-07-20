from sklearn.cluster import KMeans
import numpy as np


class DensityTransitionMatrix:

    def __init__(self, k_regions=50):

        self.k_regions = k_regions

    def compute(self, Z_old, Z_new):

        km = KMeans(
            n_clusters=self.k_regions,
            random_state=0
        )

        labels_old = km.fit_predict(Z_old)

        labels_new = km.predict(Z_new)

        M = np.zeros(
            (self.k_regions, self.k_regions)
        )

        for a, b in zip(
                labels_old,
                labels_new
        ):
            M[a, b] += 1

        M /= (
                M.sum(axis=1, keepdims=True)
                + 1e-8
        )

        return M