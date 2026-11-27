import numpy as np
import pytest

from embedding_drift_lib.analyser import DriftAnalyser
from embedding_drift_lib.config import DriftConfig


def test_mmd_score_is_symmetric_when_reference_and_current_are_swapped():
    rng = np.random.default_rng(2026)
    reference = rng.normal(size=(80, 4)) * np.array([5.0, 2.0, 0.3, 0.2])
    current = (
        rng.normal(size=(80, 4)) * np.array([0.5, 3.0, 2.0, 0.2])
        + np.array([0.0, 1.5, 0.0, 0.0])
    )
    config = DriftConfig(
        index_name="pooled-pca-regression",
        pca_components=1,
        n_permutations=19,
    )

    forward = DriftAnalyser(config).analyse(reference, current)
    reverse = DriftAnalyser(config).analyse(current, reference)

    # A two-sample MMD statistic must not depend on which sample is named
    # reference. Reference-only PCA violates this when components are truncated.
    assert forward.mmd_score == pytest.approx(reverse.mmd_score, abs=1e-6)
