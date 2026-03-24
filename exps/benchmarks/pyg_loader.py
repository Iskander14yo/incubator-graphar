from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from typing import cast

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .timings import BatchTimings


class PyGNeighborLoader:
    """PyG in-memory upper-bound baseline.

    Loads the full OGB graph into RAM once and wraps PyG's NeighborLoader.
    No internal stage breakdown is available (PyG internals are opaque).
    """

    def __init__(
        self,
        dataset_name: str,
        ogb_root: str,
        num_neighbors: list[int],
        input_nodes: torch.Tensor | None = None,
        batch_size: int = 128,
        shuffle: bool = True,
    ) -> None:
        ogb = PygNodePropPredDataset(name=dataset_name, root=ogb_root)
        self._data = cast(Data, ogb[0])
        if input_nodes is None:
            input_nodes = ogb.get_idx_split()["train"]
        self._loader = NeighborLoader(
            self._data,
            num_neighbors=num_neighbors,
            input_nodes=input_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.timings: defaultdict = defaultdict(list)

    def __iter__(self) -> Iterator[Data]:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


def iter_batches(loader: PyGNeighborLoader) -> Iterator[tuple[Data, BatchTimings]]:
    """Wraps PyGNeighborLoader with per-batch timing.

    retrieval_ms covers the full next() call; conversion_ms is 0 (PyG internals are opaque).
    """
    loader.timings = defaultdict(list)
    t = loader.timings

    gen = iter(loader)
    batch_id = 0
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(gen)
        except StopIteration:
            break
        retrieval_ms = (time.perf_counter() - t0) * 1000
        t["retrieval"].append(retrieval_ms / 1000)

        yield batch, BatchTimings(
            batch_id=batch_id,
            total_ms=retrieval_ms,
            retrieval_ms=retrieval_ms,
            conversion_ms=0.0,
            sampled_nodes=int(batch.n_id.size(0)),
            sampled_edges=int(batch.edge_index.size(1)) if batch.edge_index is not None else 0,
        )
        batch_id += 1
