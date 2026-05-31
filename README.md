# embedding-drift-lib

**Detect when your RAG vector index silently drifts from reality.**

Pure Python library — no external services, no API keys, no database required.
3 dependencies: `numpy`, `scikit-learn`, `scipy`.

## Install

```bash
pip install embedding-drift-lib
```

## Quick Start

```python
from embedding_drift_lib import DriftAnalyser, DriftConfig
import numpy as np

config   = DriftConfig(index_name="my-rag")
analyser = DriftAnalyser(config)

# Pass your index vectors and recent query vectors
report = analyser.analyse(reference_vectors, current_vectors)

print(report.drift_type)       # "none" | "mild" | "significant" | "severe"
print(report.mmd_score)        # 0.0 = identical, > 0.20 = severe drift
print(report.primary_cause)    # "model_drift" | "query_shift" | "data_staleness"
print(report.recommendation)   # human-readable action to take
```

## Wrap Your Embedding Function (Zero-Touch Monitoring)

```python
from embedding_drift_lib import DriftMonitor, DriftConfig
from openai import AsyncOpenAI

client  = AsyncOpenAI()
monitor = DriftMonitor(DriftConfig(index_name="my-rag"))

# Create baseline from your existing index (run once)
await monitor.create_snapshot(index_vectors, label="v1-baseline")

# Wrap your embed call — your RAG code does not change
embed = monitor.wrap(client.embeddings.create)

# Use normally — drift analysis runs in the background automatically
response = await embed(model="text-embedding-3-small", input=["user query"])

# Check anytime
report = monitor.last_report
```

## What It Detects

| Drift Type | Signal | Example |
|---|---|---|
| **Model drift** | High MMD + concentrated Wasserstein | OpenAI silently updates model weights |
| **Query shift** | High coverage gap + low MMD | Users ask about a new topic your corpus never covered |
| **Data staleness** | High MMD + diffuse Wasserstein | Corpus ages relative to current user queries |

## Configuration

```python
config = DriftConfig(
    index_name                   = "my-rag",
    pca_components               = 50,      # reduce embedding dims before MMD
    n_permutations               = 200,     # permutation test iterations
    sample_rate                  = 0.05,    # sample 5% of queries (default)
    elevated_sample_rate         = 0.30,    # 30% when drift suspected
    analysis_batch_size          = 200,     # analyse every N queries
    mmd_mild_threshold           = 0.02,
    mmd_significant_threshold    = 0.08,
    mmd_severe_threshold         = 0.20,
    coverage_gap_alert_threshold = 0.15,    # alert when 15%+ queries uncovered
    snapshot_dir                 = "./snapshots",
)
```

> [!TIP]
> **Tip for Local Testing:** By default, the library only monitors 5% of queries (`sample_rate = 0.05`) to protect production performance. If you are testing the library locally with a small dataset or manual queries, set `sample_rate = 1.0` in your `DriftConfig` to ensure 100% of your test queries are captured!

## How It Works

```
Your RAG App
    |
    | query -> embed() -> vector search -> LLM -> answer
    v
DriftMonitor wrapper (transparent)
    |
    +-- Samples 5% of embedding calls (adaptive: 30% when drift suspected)
    |
    +-- Every 200 samples, runs DriftAnalyser:
    |       1. PCA projection      1536-dim -> 50-dim
    |       2. MMD^2               primary drift metric
    |       3. Wasserstein         which semantic directions shifted
    |       4. Coverage gap        query shift detection
    |       5. Permutation test    200 shuffles, p < 0.05
    |
    +-- Classifies: none / mild / significant / severe
    +-- Diagnoses:  model_drift / query_shift / data_staleness
    +-- Returns DriftReport
```

## The Math

**MMD (Maximum Mean Discrepancy)** — non-parametric test measuring distance between two distributions without assuming any shape. Uses RBF kernel with median heuristic bandwidth.

**Wasserstein distance** — computed per PCA dimension. Concentration ratio (max/mean) distinguishes sudden model swap (high concentration) from gradual data staleness (diffuse).

**Coverage gap** — fraction of queries whose best cosine similarity to any index document falls below threshold. Catches query shift that MMD misses.

**Permutation test** — 200 label shuffles build a null distribution. Observed MMD must exceed the 95th percentile of the null to be reported as statistically significant.

## Author

**Sarthak Pande** — [GitHub](https://github.com/sarthakpande108)

## License

MIT
