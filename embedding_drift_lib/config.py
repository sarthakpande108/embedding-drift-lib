"""
DriftConfig — lean config for the library.

No Supabase, no alerts, no OpenAI credentials.
Pure logic parameters only.
"""

from dataclasses import dataclass, field


@dataclass
class DriftConfig:
    # Identity
    index_name: str = "default"

    # Math
    pca_components: int = 50
    n_permutations: int = 200
    significance_alpha: float = 0.05

    # Sampling
    sample_rate: float = 0.05
    elevated_sample_rate: float = 0.30
    analysis_batch_size: int = 200
    max_query_buffer_size: int = 2000
    reservoir_size: int = 2000
    n_clusters: int = 20

    # Drift severity thresholds (tune after 2 weeks of data)
    mmd_mild_threshold: float = 0.02
    mmd_significant_threshold: float = 0.08
    mmd_severe_threshold: float = 0.20

    # Coverage thresholds
    coverage_gap_alert_threshold: float = 0.15   # fraction of uncovered queries → alert
    coverage_similarity_threshold: float = 0.65  # min cosine sim to count as "covered"

    # Snapshot storage
    snapshot_dir: str = "./snapshots"

    def __post_init__(self):
        assert 0.0 <= self.sample_rate <= 1.0, "sample_rate must be in [0, 1]"
        assert 0.0 <= self.elevated_sample_rate <= 1.0, "elevated_sample_rate must be in [0, 1]"
        assert self.pca_components > 0, "pca_components must be positive"
        assert self.n_permutations > 0, "n_permutations must be positive"

        for name, value in (
            ("analysis_batch_size", self.analysis_batch_size),
            ("max_query_buffer_size", self.max_query_buffer_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_query_buffer_size < self.analysis_batch_size:
            raise ValueError(
                "max_query_buffer_size must be greater than or equal to analysis_batch_size"
            )
