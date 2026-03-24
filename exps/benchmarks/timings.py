from __future__ import annotations

from dataclasses import dataclass, field

from graphar.ml.torch import _Timer as _Timer  # re-exported; defined in torch.py to avoid reverse dep


@dataclass
class BatchTimings:
    batch_id: int
    total_ms: float
    retrieval_ms: float
    conversion_ms: float
    sampled_nodes: int
    sampled_edges: int
    # GAR-specific sub-stages (None for other loaders)
    sampling_ms: float | None = field(default=None)
    feature_fetch_ms: float | None = field(default=None)
    # Neo4j-specific (None for other loaders; populated every profile_every_n batches)
    neo4j_profile: dict | None = field(default=None)
