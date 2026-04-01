from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator

from torch_geometric.data import Data

from graphar.ml.torch import GARNeighborLoader

from .timings import BatchTimings


def iter_batches(loader: GARNeighborLoader) -> Iterator[tuple[Data, BatchTimings]]:
    loader.timings = defaultdict(list)
    t = loader.timings

    gen = iter(loader)
    batch_id = 0
    while batch_id < 50:
        t_start = time.perf_counter()
        try:
            batch = next(gen)
        except StopIteration:
            break
        total_ms = (time.perf_counter() - t_start) * 1000

        sampling_ms = t["sampling"][-1] * 1000
        feature_fetch_ms = t["feature_fetch"][-1] * 1000 if t["feature_fetch"] else 0.0
        conversion_ms = t["conversion"][-1] * 1000 if t["conversion"] else 0.0

        yield batch, BatchTimings(
            batch_id=batch_id,
            total_ms=total_ms,
            retrieval_ms=sampling_ms + feature_fetch_ms,
            conversion_ms=conversion_ms,
            sampling_ms=sampling_ms,
            feature_fetch_ms=feature_fetch_ms,
            sampled_nodes=int(batch.n_id.size(0)),
            sampled_edges=int(batch.edge_index.size(1)) if batch.edge_index is not None else 0,
        )
        batch_id += 1
