import numpy as np


class SpectralAnalyzer:

    def compute(self, Z):

        C = np.cov(Z.T)

        eigvals = np.linalg.eigvalsh(C)

        eigvals = np.maximum(eigvals, 0)

        deff = (
                       eigvals.sum() ** 2
               ) / (
                       (eigvals ** 2).sum() + 1e-8
               )

        return {
            "effective_dim": deff,
            "eigvals": eigvals
        }