"""
embedding-drift-lib — Usage Examples

Shows 3 ways to use the library:
  1. One-shot analysis (no wrapping, just pass vectors directly)
  2. Wrap your embedding function (automatic background monitoring)
  3. Force manual analysis check

No API keys, no Supabase, no Slack — pure logic.
Run: python example_usage.py
"""

import asyncio
import numpy as np
from embedding_drift_lib import DriftMonitor, DriftConfig, DriftAnalyser, DriftReport


# -- Helper: simulate embedding vectors ----------------------------------------
# In real usage these come from: openai client.embeddings.create(...)
# Here we generate fake 1536-dim vectors (same shape as text-embedding-3-small)

def fake_embed(texts: list[str], mean: float = 0.0) -> np.ndarray:
    """Simulate OpenAI-style embeddings (1536-dim float32)."""
    rng = np.random.default_rng(seed=abs(hash(str(texts))) % 2**32)
    vecs = rng.normal(mean, 1.0, (len(texts), 1536)).astype(np.float32)
    # Normalize to unit sphere (like real embeddings)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ==============================================================================
# EXAMPLE 1 — One-shot analysis (simplest usage)
# Pass your reference vectors and current vectors directly.
# No wrapping, no snapshot files, just instant results.
# ==============================================================================

def example_1_oneshot_analysis():
    print("\n" + "=" * 60)
    print("EXAMPLE 1 -- One-shot drift analysis")
    print("=" * 60)

    config = DriftConfig(
        index_name  = "my-rag-index",
        pca_components  = 50,    # reduce 1536 → 50 dims for MMD
        n_permutations  = 100,   # permutation test iterations
    )
    analyser = DriftAnalyser(config)

    # --- Scenario A: healthy (same distribution) ---
    index_vectors  = fake_embed([f"doc {i}" for i in range(300)])   # your corpus
    query_vectors  = fake_embed([f"query {i}" for i in range(100)]) # recent queries

    report = analyser.analyse(index_vectors, query_vectors)

    print(f"\n[Scenario A — Healthy]")
    print(f"  Drift type : {report.drift_type}")
    print(f"  MMD score  : {report.mmd_score:.4f}")
    print(f"  p-value    : {report.mmd_p_value:.3f}")
    print(f"  Cause      : {report.primary_cause}")
    print(f"  Action     : {report.recommendation}")

    # --- Scenario B: severe drift (model updated silently) ---
    # Simulated by shifting mean by 8 — equivalent to a model swap
    drifted_queries = fake_embed([f"query {i}" for i in range(100)], mean=8.0)

    # Reuse the same analyser — projector already fitted on index_vectors
    report_drift = analyser.analyse(index_vectors, drifted_queries)

    print(f"\n[Scenario B — Severe Drift]")
    print(f"  Drift type : {report_drift.drift_type}")
    print(f"  MMD score  : {report_drift.mmd_score:.4f}")
    print(f"  p-value    : {report_drift.mmd_p_value:.3f}")
    print(f"  Cause      : {report_drift.primary_cause}")
    print(f"  Alert      : {report_drift.alert_severity}")
    print(f"  Action     : {report_drift.recommendation[:80]}...")

    # Full dict output (ready to log / send to your DB)
    d = report_drift.to_dict()
    print(f"\n  to_dict() keys: {list(d.keys())}")


# ==============================================================================
# EXAMPLE 2 — Wrap your embedding function (real RAG integration)
# The monitor wraps your embed function. Every call is intercepted,
# vectors are buffered, analysis fires automatically in the background.
# ==============================================================================

async def example_2_wrap_embedding_function():
    print("\n" + "=" * 60)
    print("EXAMPLE 2 — Wrap embedding function (RAG integration)")
    print("=" * 60)

    import tempfile, os

    config = DriftConfig(
        index_name          = "production-rag",
        pca_components      = 50,
        n_permutations      = 50,    # lower for speed in this demo
        analysis_batch_size = 60,    # trigger analysis every 60 queries
        sample_rate         = 1.0,   # sample 100% for demo (use 0.05 in prod)
        snapshot_dir        = tempfile.mkdtemp(),
    )
    monitor = DriftMonitor(config)

    # -- Step 1: Build your index and create a baseline snapshot -----------
    # In real usage: these are the vectors already in your vector DB
    print("\n[Step 1] Creating baseline snapshot from index vectors...")
    index_docs = [f"document about topic {i}" for i in range(300)]
    index_vectors = fake_embed(index_docs)

    snapshot_id = await monitor.create_snapshot(index_vectors, label="v1-baseline")
    print(f"  Snapshot ID : {snapshot_id}")
    print(f"  Vectors     : {len(index_vectors)} x {index_vectors.shape[1]}-dim")

    # -- Step 2: Wrap your embedding function ------------------------------
    # In real usage: wrap client.embeddings.create from openai or azure
    async def my_embed_function(texts: list[str], mean: float = 0.0):
        """Simulates async OpenAI embeddings.create() response."""
        class EmbeddingItem:
            def __init__(self, vec): self.embedding = vec.tolist()
        class EmbeddingResponse:
            def __init__(self, vecs): self.data = [EmbeddingItem(v) for v in vecs]
        vecs = fake_embed(texts, mean=mean)
        return EmbeddingResponse(vecs)

    # One line to enable monitoring — your RAG code doesn't change
    embed = monitor.wrap(my_embed_function)

    # -- Step 3: Simulate normal queries (no drift expected) ---------------
    print("\n[Step 2] Running 70 normal queries (no drift expected)...")
    for i in range(70):
        await embed([f"normal query {i}"])   # monitor samples these silently

    # Trigger analysis manually since batch_size=60 already fired
    report = await monitor.force_analysis()
    if report:
        print(f"  Drift type : {report.drift_type}")
        print(f"  MMD score  : {report.mmd_score:.4f}")
        print(f"  Alert      : {report.requires_alert}")
    else:
        print("  No report yet (buffer empty after auto-analysis)")

    # -- Step 4: Simulate model update (severe drift) ----------------------
    print("\n[Step 3] Simulating silent model update (severe drift)...")
    # In real usage: OpenAI updated the model weights — new embeddings
    # live in a completely different region of the vector space
    for i in range(70):
        await embed([f"same query {i}"], mean=8.0)   # shifted distribution

    report = await monitor.force_analysis()
    if report:
        print(f"  Drift type : {report.drift_type}")
        print(f"  MMD score  : {report.mmd_score:.4f}  (> 0.20 = severe)")
        print(f"  Cause      : {report.primary_cause}")
        print(f"  Alert      : {report.alert_severity}")
        print(f"  Action     : {report.recommendation[:80]}...")

    # -- Access last report anytime -----------------------------------------
    print(f"\n  monitor.last_report.drift_type = {monitor.last_report.drift_type}")
    print(f"  monitor.current_sample_size   = {monitor.current_sample_size}")


# ==============================================================================
# EXAMPLE 3 — Custom thresholds (tune for your domain)
# Medical/legal RAG needs tighter thresholds than general-purpose.
# All thresholds live in DriftConfig — no code changes needed.
# ==============================================================================

def example_3_custom_thresholds():
    print("\n" + "=" * 60)
    print("EXAMPLE 3 — Custom thresholds (medical/legal RAG)")
    print("=" * 60)

    # Tighter thresholds — catch drift earlier in high-stakes domains
    config = DriftConfig(
        index_name                  = "medical-rag",
        mmd_mild_threshold          = 0.01,   # default: 0.02
        mmd_significant_threshold   = 0.04,   # default: 0.08
        mmd_severe_threshold        = 0.10,   # default: 0.20
        coverage_gap_alert_threshold= 0.05,   # default: 0.15  (alert at 5% not 15%)
        coverage_similarity_threshold= 0.80,  # default: 0.65  (stricter "covered" check)
        n_permutations              = 500,    # more permutations = more reliable p-value
        pca_components              = 100,    # more PCA components = more detail
    )
    analyser = DriftAnalyser(config)

    index_vectors  = fake_embed([f"medical doc {i}" for i in range(300)])
    query_vectors  = fake_embed([f"patient query {i}" for i in range(100)], mean=0.5)

    report = analyser.analyse(index_vectors, query_vectors)

    print(f"\n  Drift type : {report.drift_type}  (tighter thresholds = earlier detection)")
    print(f"  MMD score  : {report.mmd_score:.4f}")
    print(f"  Alert      : {report.requires_alert}")
    print(f"  Cause      : {report.primary_cause}")


# ==============================================================================
# EXAMPLE 4 — Real OpenAI integration (uncomment when you have a key)
# ==============================================================================

async def example_4_real_openai():
    """
    Uncomment and run with a real OpenAI key to see live drift detection.

    pip install openai
    export OPENAI_API_KEY=sk-...
    """
    # import os
    # from openai import AsyncOpenAI
    #
    # client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    #
    # config = DriftConfig(index_name="openai-rag", snapshot_dir="./snapshots")
    # monitor = DriftMonitor(config)
    #
    # # Create snapshot from your corpus
    # corpus = ["What is machine learning?", "How does RAG work?", ...]
    # response = await client.embeddings.create(model="text-embedding-3-small", input=corpus)
    # corpus_vectors = np.array([item.embedding for item in response.data])
    # await monitor.create_snapshot(corpus_vectors, label="v1")
    #
    # # Wrap and monitor
    # embed = monitor.wrap(client.embeddings.create)
    #
    # # Normal usage — nothing changes in your RAG code
    # result = await embed(model="text-embedding-3-small", input=["user query here"])
    # vectors = np.array([item.embedding for item in result.data])  # use as normal
    #
    # report = monitor.last_report
    # if report:
    #     print(f"Drift: {report.drift_type}, MMD: {report.mmd_score:.4f}")

    print("\n[Example 4] — Commented out. Uncomment with your OPENAI_API_KEY.")


# -- Run all examples -----------------------------------------------------------

async def main():
    example_1_oneshot_analysis()
    await example_2_wrap_embedding_function()
    example_3_custom_thresholds()
    await example_4_real_openai()
    print("\n" + "=" * 60)
    print("All examples complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())