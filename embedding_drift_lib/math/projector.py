"""
PCA projector — reduces embedding vectors before computing drift metrics.

Why PCA and not UMAP/t-SNE?
  - PCA is deterministic. Same input → same output every time.
  - PCA is fast (milliseconds for 2000 vectors).
  - PCA is invertible — you can reason about what changed in the original space.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
from sklearn.decomposition import PCA


def _as_finite_matrix(vectors: np.ndarray, name: str) -> np.ndarray:
    """Convert array-like input and reject non-numeric or non-finite matrices."""
    try:
        matrix = np.asarray(vectors)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric 2D array") from exc

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D: (n_samples, embedding_dim)")
    if not np.issubdtype(matrix.dtype, np.number) or np.iscomplexobj(matrix):
        raise ValueError(f"{name} must contain real numeric values")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


class DriftProjector:
    """Wraps sklearn PCA with persistence and validation."""

    def __init__(self, n_components: int = 50):
        if (
            isinstance(n_components, (bool, np.bool_))
            or not isinstance(n_components, (int, np.integer))
            or n_components <= 0
        ):
            raise ValueError("n_components must be a positive integer")

        self.n_components = int(n_components)
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.fitted = False
        self.reference_dim: int | None = None
        self.explained_variance_ratio: float | None = None

    def fit(self, reference_vectors: np.ndarray) -> DriftProjector:
        """
        Fit on a finite reference snapshot.

        PCA requires at least two observations and cannot produce more
        components than either the number of observations or input features.
        """
        reference_vectors = _as_finite_matrix(reference_vectors, "reference_vectors")
        n_samples, n_features = reference_vectors.shape
        if n_samples < 2:
            raise ValueError("reference_vectors must contain at least two samples")
        if n_features == 0:
            raise ValueError("reference_vectors must contain at least one feature")

        effective_components = min(self.n_components, n_samples, n_features)
        candidate = PCA(n_components=effective_components, random_state=42)
        candidate.fit(reference_vectors)

        # Publish the new estimator only after fitting succeeds. A failed refit
        # must not leave a previously usable projector partially mutated.
        self.pca = candidate
        self.fitted = True
        self.reference_dim = n_features
        self.explained_variance_ratio = float(candidate.explained_variance_ratio_.sum())
        return self

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """
        Project vectors into the PCA space fitted on reference.

        Args:
            vectors: Shape (n_samples, embedding_dim).
        """
        if not self.fitted:
            raise RuntimeError("Call fit() with reference vectors before projecting.")

        vectors = _as_finite_matrix(vectors, "vectors")
        if vectors.shape[0] == 0:
            raise ValueError("vectors must contain at least one sample")
        if vectors.shape[1] != self.reference_dim:
            raise ValueError(
                f"Embedding dim mismatch: projector was fit on dim={self.reference_dim}, "
                f"got dim={vectors.shape[1]}. Did you change embedding models?"
            )

        return self.pca.transform(vectors)

    def save(self, path: str) -> None:
        """Persist projector so you don't need to re-fit after restart."""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "pca": self.pca,
                "fitted": self.fitted,
                "reference_dim": self.reference_dim,
                "n_components": self.n_components,
                "explained_variance_ratio": self.explained_variance_ratio,
            }, f)

    @classmethod
    def load(cls, path: str) -> DriftProjector:
        """Load a previously saved projector."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        proj = cls(n_components=data["n_components"])
        proj.pca = data["pca"]
        proj.fitted = data["fitted"]
        proj.reference_dim = data["reference_dim"]
        proj.explained_variance_ratio = data["explained_variance_ratio"]
        return proj

    @property
    def info(self) -> dict:
        return {
            "n_components": self.n_components,
            "effective_n_components": (
                int(self.pca.n_components_) if self.fitted else None
            ),
            "reference_dim": self.reference_dim,
            "fitted": self.fitted,
            "explained_variance_ratio": self.explained_variance_ratio,
        }
