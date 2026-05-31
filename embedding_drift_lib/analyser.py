"""
DriftAnalyser — orchestrates all math into a single DriftReport.

No external dependencies (no Supabase, no alerts, no OpenAI).
Pure analysis: reference vectors + current vectors → DriftReport.
"""

from dataclasses import dataclass, field
from typing import Optional
import datetime
import numpy as np

from .config import DriftConfig
from .math import (
    compute_mmd,
    is_drift_significant,
    compute_wasserstein_per_dim,
    compute_coverage_gap,
    DriftProjector,
)


@dataclass
class DriftReport:
    """Complete drift analysis result for one analysis run."""

    timestamp: str
    index_name: str

    mmd_score: float
    mmd_p_value: float
    is_statistically_significant: bool

    wasserstein_mean: float
    wasserstein_concentration: float
    wasserstein_top_dims: list[tuple[int, float]]

    coverage_gap_rate: float
    mean_retrieval_similarity: float

    drift_type: str        # "none" | "mild" | "significant" | "severe"
    primary_cause: str     # "model_drift" | "query_shift" | "data_staleness" | "unknown"
    recommendation: str

    requires_alert: bool
    alert_severity: Optional[str]  # None | "warning" | "critical"

    reference_sample_size: int
    current_sample_size: int

    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp":                 self.timestamp,
            "index_name":                self.index_name,
            "mmd_score":                 self.mmd_score,
            "mmd_p_value":               self.mmd_p_value,
            "is_significant":            self.is_statistically_significant,
            "wasserstein_mean":          self.wasserstein_mean,
            "wasserstein_concentration": self.wasserstein_concentration,
            "coverage_gap_rate":         self.coverage_gap_rate,
            "mean_retrieval_similarity": self.mean_retrieval_similarity,
            "drift_type":                self.drift_type,
            "primary_cause":             self.primary_cause,
            "recommendation":            self.recommendation,
            "requires_alert":            self.requires_alert,
            "alert_severity":            self.alert_severity,
            "reference_sample_size":     self.reference_sample_size,
            "current_sample_size":       self.current_sample_size,
        }


class DriftAnalyser:
    """
    Runs all drift metrics and produces a DriftReport.

    Usage:
        from embedding_drift_lib import DriftAnalyser, DriftConfig
        analyser = DriftAnalyser(DriftConfig(index_name="my-rag"))
        report = analyser.analyse(reference_vectors, current_vectors)
    """

    def __init__(self, config: DriftConfig):
        self.config = config
        self.projector = DriftProjector(n_components=config.pca_components)
        # All thresholds from config — fully tunable
        self._mmd_thresholds = {
            "mild":        config.mmd_mild_threshold,
            "significant": config.mmd_significant_threshold,
            "severe":      config.mmd_severe_threshold,
        }

    def analyse(self, reference: np.ndarray, current: np.ndarray) -> DriftReport:
        """
        Full drift analysis pipeline.

        Args:
            reference: Vectors from reference snapshot, shape (n_ref, dim)
            current:   Recent query/index vectors,      shape (n_cur, dim)

        Returns:
            DriftReport with severity, cause, and recommendation.
        """
        reference = np.asarray(reference, dtype=np.float32)
        current   = np.asarray(current,   dtype=np.float32)

        if reference.ndim != 2 or current.ndim != 2:
            raise ValueError("Both reference and current must be 2D arrays (n_samples, dim)")
        if reference.shape[1] != current.shape[1]:
            raise ValueError(
                f"Dimension mismatch: reference={reference.shape[1]}, current={current.shape[1]}"
            )

        if not self.projector.fitted:
            self.projector.fit(reference)

        ref_proj = self.projector.project(reference)
        cur_proj = self.projector.project(current)

        significance = is_drift_significant(
            ref_proj, cur_proj,
            n_permutations=self.config.n_permutations,
            alpha=self.config.significance_alpha,
        )
        w_scores = compute_wasserstein_per_dim(ref_proj, cur_proj)
        coverage = compute_coverage_gap(
            reference, current,
            threshold=self.config.coverage_similarity_threshold,
        )

        mmd = significance["observed_mmd"]

        if mmd >= self._mmd_thresholds["severe"]:
            drift_type = "severe"
        elif mmd >= self._mmd_thresholds["significant"]:
            drift_type = "significant"
        elif mmd >= self._mmd_thresholds["mild"]:
            drift_type = "mild"
        else:
            drift_type = "none"

        cause          = self._diagnose_cause(mmd, coverage, w_scores)
        recommendation = self._make_recommendation(drift_type, cause, coverage)

        alert_severity = None
        requires_alert = False

        if significance["is_significant"]:
            if drift_type == "severe":
                requires_alert = True
                alert_severity = "critical"
            elif drift_type in ("significant", "mild"):
                requires_alert = True
                alert_severity = "warning"

        if coverage["coverage_gap_rate"] > self.config.coverage_gap_alert_threshold:
            requires_alert = True
            alert_severity = alert_severity or "warning"

        return DriftReport(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            index_name=self.config.index_name,
            mmd_score=mmd,
            mmd_p_value=significance["p_value"],
            is_statistically_significant=significance["is_significant"],
            wasserstein_mean=w_scores["mean"],
            wasserstein_concentration=w_scores["concentration"],
            wasserstein_top_dims=w_scores["top_drifted_dims"],
            coverage_gap_rate=coverage["coverage_gap_rate"],
            mean_retrieval_similarity=coverage["mean_best_similarity"],
            drift_type=drift_type,
            primary_cause=cause,
            recommendation=recommendation,
            requires_alert=requires_alert,
            alert_severity=alert_severity,
            reference_sample_size=len(reference),
            current_sample_size=len(current),
            raw={"significance": significance, "wasserstein": w_scores, "coverage": coverage},
        )

    def _diagnose_cause(self, mmd: float, coverage: dict, w_scores: dict) -> str:
        high_mmd          = mmd > self._mmd_thresholds["mild"]
        high_coverage_gap = coverage["coverage_gap_rate"] > self.config.coverage_gap_alert_threshold
        concentrated       = w_scores["concentration"] > 3.0

        if high_mmd and concentrated:
            return "model_drift"
        if high_coverage_gap and not high_mmd:
            return "query_shift"
        if high_mmd and not concentrated:
            return "data_staleness"
        return "unknown"

    def _make_recommendation(self, drift_type: str, cause: str, coverage: dict) -> str:
        if drift_type == "none":
            return "Index is healthy. No action required."

        recommendations = {
            "model_drift": (
                "Embedding model may have been updated silently. "
                "Compare your current model version against your index creation date. "
                "If confirmed, re-index your entire corpus with the current model."
            ),
            "query_shift": (
                f"{coverage['coverage_gap_rate'] * 100:.1f}% of recent queries have no "
                "strong match in your index. Your users are asking about topics your "
                "corpus does not cover. Cluster the uncovered queries to identify new "
                "topics, then add documents for those areas."
            ),
            "data_staleness": (
                "Your corpus distribution has gradually diverged from current query "
                "patterns. Review and refresh documents older than 6 months, "
                "then run a full re-index."
            ),
            "unknown": (
                "Drift detected but cause unclear. Manually review a sample of recent "
                "queries vs. retrieved chunks. Check the Wasserstein breakdown for "
                "which semantic dimensions shifted most."
            ),
        }
        return recommendations.get(cause, recommendations["unknown"])