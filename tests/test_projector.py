import numpy as np
import pytest

from embedding_drift_lib.math.projector import DriftProjector


def test_fit_caps_components_to_embedding_dimension():
    reference = np.random.default_rng(1).normal(size=(30, 3))
    projector = DriftProjector(n_components=50).fit(reference)

    assert projector.pca.n_components_ == 3
    assert projector.project(reference).shape == (30, 3)
    assert projector.info["effective_n_components"] == 3


def test_fit_caps_components_to_reference_sample_count():
    reference = np.random.default_rng(2).normal(size=(3, 8))
    projector = DriftProjector(n_components=5).fit(reference)

    assert projector.pca.n_components_ == 3
    assert projector.project(reference).shape == (3, 3)


def test_fit_keeps_requested_components_when_supported():
    reference = np.random.default_rng(3).normal(size=(20, 8))
    projector = DriftProjector(n_components=4).fit(reference)

    assert projector.pca.n_components_ == 4
    assert projector.info["n_components"] == 4
    assert projector.project(reference).shape == (20, 4)


@pytest.mark.parametrize("n_components", [0, -1, 1.5, True])
def test_constructor_rejects_invalid_component_counts(n_components):
    with pytest.raises(ValueError, match="positive integer"):
        DriftProjector(n_components=n_components)


@pytest.mark.parametrize(
    "reference, message",
    [
        (np.array([1.0, 2.0]), "must be 2D"),
        (np.empty((0, 3)), "at least two samples"),
        (np.ones((1, 3)), "at least two samples"),
        (np.empty((3, 0)), "at least one feature"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), "finite values"),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), "finite values"),
        (np.array([["one", "two"], ["three", "four"]]), "real numeric"),
    ],
)
def test_fit_rejects_invalid_reference_matrices(reference, message):
    with pytest.raises(ValueError, match=message):
        DriftProjector().fit(reference)


def test_failed_refit_preserves_previous_fitted_projector():
    reference = np.random.default_rng(4).normal(size=(20, 4))
    projector = DriftProjector(n_components=3).fit(reference)
    expected = projector.project(reference)

    with pytest.raises(ValueError, match="at least two samples"):
        projector.fit(np.ones((1, 4)))

    np.testing.assert_allclose(projector.project(reference), expected)


@pytest.mark.parametrize(
    "vectors, message",
    [
        (np.array([1.0, 2.0]), "must be 2D"),
        (np.empty((0, 4)), "at least one sample"),
        (np.ones((2, 5)), "Embedding dim mismatch"),
        (np.array([[np.nan, 1.0, 2.0, 3.0]]), "finite values"),
        (np.array([["one", "two", "three", "four"]]), "real numeric"),
    ],
)
def test_project_rejects_invalid_current_matrices(vectors, message):
    reference = np.random.default_rng(5).normal(size=(20, 4))
    projector = DriftProjector(n_components=2).fit(reference)

    with pytest.raises(ValueError, match=message):
        projector.project(vectors)


def test_project_requires_a_fitted_reference():
    with pytest.raises(RuntimeError, match="Call fit"):
        DriftProjector().project(np.ones((2, 3)))


def test_save_and_load_preserve_effective_projection(tmp_path):
    reference = np.random.default_rng(6).normal(size=(25, 3))
    projector = DriftProjector(n_components=50).fit(reference)
    path = tmp_path / "nested" / "projector.pkl"

    projector.save(str(path))
    restored = DriftProjector.load(str(path))

    assert restored.info["effective_n_components"] == 3
    np.testing.assert_allclose(
        restored.project(reference), projector.project(reference)
    )
