"""
PCA projector — reduces 1536-dim vectors to 50-dim before computing MMD.

Why PCA and not UMAP/t-SNE?
  - PCA is deterministic. Same input → same output every time.
  - PCA is fast (milliseconds for 2000 vectors).
  - PCA is invertible — you can reason about what changed in the original space.
"""

from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os


class DriftProjector:
    """
    Wraps sklearn PCA with persistence and validation.
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.fitted = False
        self.reference_dim: int | None = None
        self.explained_variance_ratio: float | None = None

    def fit(self, reference_vectors: np.ndarray) -> DriftProjector:
        """
        Fit on reference snapshot. Call ONCE, then reuse.

        Args:
            reference_vectors: Shape (n_samples, embedding_dim).
        """
        if reference_vectors.ndim != 2:
            raise ValueError("reference_vectors must be 2D: (n_samples, embedding_dim)")
        if len(reference_vectors) < self.n_components:
            # Gracefully downgrade for small testing datasets so we don't crash
            self.pca.n_components = len(reference_vectors)
        else:
            self.pca.n_components = self.n_components

        self.pca.fit(reference_vectors)
        self.fitted = True
        self.reference_dim = reference_vectors.shape[1]
        self.explained_variance_ratio = float(
            self.pca.explained_variance_ratio_.sum()
        )
        return self

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """
        Project vectors into the PCA space fitted on reference.

        Args:
            vectors: Shape (n_samples, embedding_dim).
        """
        if not self.fitted:
            raise RuntimeError("Call fit() with reference vectors before projecting.")
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
            "reference_dim": self.reference_dim,
            "fitted": self.fitted,
            "explained_variance_ratio": self.explained_variance_ratio,
        }