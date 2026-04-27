from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import pyarrow as pa
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data

import graphar.ml as gar_ml


@dataclass
class BatchProfile:
    """Per-batch timing breakdown produced by GARNeighborLoader.profile()."""
    total_ms: float
    sampling_ms: float
    feature_fetch_ms: float
    conversion_ms: float


def _chunked_array_to_tensor(column: pa.ChunkedArray, name: str) -> torch.Tensor:
    combined = column.combine_chunks()
    numpy_array = combined.to_numpy(zero_copy_only=False)
    # it's possible to validate types on initialization:
    # `graph_info.get_vertex_info().get_property_type(column).to_type_name()`
    # but it requires hardcoding all numeric GAR types as strings making validation brittle
    # therefore try-except is used
    # no performance penalty is expected since exception isn't supposed to be raised
    try:
        # from_numpy may warn if underlying array is read-only; this loader is read-only so it's ok
        tensor = torch.from_numpy(numpy_array)
    except TypeError:
        msg = f"Feature '{name}' cannot be converted to a tensor (non-numeric type: {column.type})"
        raise TypeError(msg) from None
    return tensor.to(torch.float32)


def _table_to_feature_tensor(table: pa.Table) -> torch.Tensor:
    """Convert pyarrow Table (from cpp binding) to pytorch Tensor"""
    if table.num_columns == 0:
        return torch.empty((table.num_rows, 0), dtype=torch.float32)
    columns = [_chunked_array_to_tensor(table.column(i), table.schema.field(i).name) for i in range(table.num_columns)]
    return torch.stack(columns, dim=1)


def _properties_for_vertex(graph_info, vertex_type: str) -> list[str]:
    """Collect all properties when features=None"""
    vertex_info = graph_info.get_vertex_info(vertex_type)
    if vertex_info is None:
        msg = f"Vertex type '{vertex_type}' not found"
        raise ValueError(msg)
    properties: list[str] = []
    for group in vertex_info.get_property_groups():
        for prop in group.get_properties():
            properties.append(prop.name)
    return properties


def _normalize_input_nodes(
    graph_info,
    vertex_type: str,
    input_nodes: Sequence[int] | torch.Tensor | None,
) -> list[int]:
    """For 3 input shapes: None / bool-mask / int-list"""
    if input_nodes is None:
        return list(range(graph_info.get_vertex_count(vertex_type)))
    if isinstance(input_nodes, torch.Tensor):
        if input_nodes.dtype == torch.bool:
            return input_nodes.nonzero(as_tuple=False).view(-1).tolist()
        return input_nodes.view(-1).to(dtype=torch.long).tolist()
    if len(input_nodes) == 0:
        return []
    if all(isinstance(node, bool) for node in input_nodes):
        return [idx for idx, keep in enumerate(input_nodes) if keep]
    return [int(node) for node in input_nodes]


def _chunk_read_stats_to_dict(stats) -> dict[str, int]:
    return {
        "requests": int(stats.requests),
        "leaders": int(stats.leaders),
        "waiters": int(stats.waiters),
        "completed": int(stats.completed),
        "failed": int(stats.failed),
        "ram_cache_hits": int(stats.ram_cache_hits),
        "ram_cache_misses": int(stats.ram_cache_misses),
        "ram_cache_evictions": int(stats.ram_cache_evictions),
        "ram_cache_bytes": int(stats.ram_cache_bytes),
    }



class GARNeighborLoader(IterableDataset):
    """Minimal PyG-compatible neighbor loader over GraphAr APIs."""

    def __init__(
        self,
        graph_info,
        vertex_type: str,
        edge_type: str,
        num_neighbors: list[int],
        input_nodes: list[int] | torch.Tensor | None = None,
        batch_size: int = 128,
        shuffle: bool = True,
        features: list[str] | None = None,
        num_workers: int = 0,
        edge_ram_for_loader_mb: int = 0,
        feature_ram_for_loader_mb: int = 0,
        feature_cursor_count: int = 1,
        feature_cursor_trail_chunks: int = 10,
    ) -> None:
        if batch_size <= 0:
            msg = "batch_size must be > 0"
            raise ValueError(msg)
        if not num_neighbors:
            msg = "num_neighbors must not be empty"
            raise ValueError(msg)
        if num_workers < 0:
            msg = "num_workers must be >= 0"
            raise ValueError(msg)
        if edge_ram_for_loader_mb < 0:
            msg = "edge_ram_for_loader_mb must be >= 0"
            raise ValueError(msg)
        if feature_ram_for_loader_mb < 0:
            msg = "feature_ram_for_loader_mb must be >= 0"
            raise ValueError(msg)
        if feature_cursor_count <= 0:
            msg = "feature_cursor_count must be > 0"
            raise ValueError(msg)
        if feature_cursor_trail_chunks < 0:
            msg = "feature_cursor_trail_chunks must be >= 0"
            raise ValueError(msg)
        self.graph_info = graph_info
        self.vertex_type = vertex_type
        self.edge_type = edge_type
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        edge_chunk_manager_options = gar_ml._ChunkReadManagerOptions()
        edge_chunk_manager_options.ram_budget_bytes = int(edge_ram_for_loader_mb) * 1024 * 1024
        self._sampling_chunk_manager = gar_ml._ChunkReadManager(edge_chunk_manager_options)
        feature_chunk_manager_options = gar_ml._ChunkReadManagerOptions()
        feature_chunk_manager_options.ram_budget_bytes = int(feature_ram_for_loader_mb) * 1024 * 1024
        self._feature_chunk_manager = gar_ml._ChunkReadManager(feature_chunk_manager_options)
        feature_cursor_options = gar_ml._FeatureCursorOptions()
        feature_cursor_options.cursor_count = int(feature_cursor_count)
        feature_cursor_options.trail_capacity_chunks = int(feature_cursor_trail_chunks)
        self._feature_coordinator = gar_ml._FeatureScanCoordinator(
            graph_info,
            self._feature_chunk_manager,
            feature_cursor_options,
        )
        if input_nodes is None and not shuffle:
            self._input_nodes = None
            self._input_node_count = graph_info.get_vertex_count(vertex_type)
        else:
            self._input_nodes = list(dict.fromkeys(  # dict.fromkeys preserves insertion order
                _normalize_input_nodes(graph_info, vertex_type, input_nodes)
            ))
            self._input_node_count = len(self._input_nodes)
        self.features = _properties_for_vertex(graph_info, vertex_type) if features is None else features
        self._rng = torch.Generator()  # used for both dataset shuffling and sampling seeds
        self._rng.manual_seed(int(torch.initial_seed()))

    def __len__(self) -> int:
        if self._input_node_count == 0:
            return 0
        return (self._input_node_count + self.batch_size - 1) // self.batch_size

    def _iter_input_batches(self) -> Iterator[list[int]]:
        total = self._input_node_count
        if total == 0:
            return
        if self._input_nodes is None:
            for start in range(0, total, self.batch_size):
                end = min(start + self.batch_size, total)
                yield list(range(start, end))
            return
        if self.shuffle:
            order = torch.randperm(total, generator=self._rng).tolist()
            ordered_nodes = [self._input_nodes[i] for i in order]
        else:
            ordered_nodes = self._input_nodes
        for start in range(0, total, self.batch_size):
            yield ordered_nodes[start : start + self.batch_size]

    def _sample_seed(self) -> int:
        return int(
            torch.randint(2**32, (1,), generator=self._rng, dtype=torch.int64).item()
        )

    def chunk_manager_stats(self) -> dict[str, int]:
        return self.sampling_chunk_manager_stats()

    def sampling_chunk_manager_stats(self) -> dict[str, int]:
        return _chunk_read_stats_to_dict(self._sampling_chunk_manager.stats())

    def feature_chunk_manager_stats(self) -> dict[str, int]:
        return _chunk_read_stats_to_dict(self._feature_chunk_manager.stats())

    def feature_cursor_stats(self) -> dict[str, int]:
        stats = self._feature_coordinator.stats()
        return {
            "cursor_count": int(stats.cursor_count),
            "trail_capacity_chunks": int(stats.trail_capacity_chunks),
            "requests": int(stats.requests),
            "requests_completed": int(stats.requests_completed),
            "requests_failed": int(stats.requests_failed),
            "active_requests_peak": int(stats.active_requests_peak),
            "chunks_read": int(stats.chunks_read),
            "chunks_served": int(stats.chunks_served),
            "rows_served": int(stats.rows_served),
            "batches_served": int(stats.batches_served),
            "trail_hits": int(stats.trail_hits),
            "trail_misses": int(stats.trail_misses),
            "trail_evictions": int(stats.trail_evictions),
            "wait_ms_sum": int(stats.wait_ms_sum),
            "wait_ms_max": int(stats.wait_ms_max),
            "service_ms_sum": int(stats.service_ms_sum),
            "service_ms_max": int(stats.service_ms_max),
        }

    def close(self) -> None:
        self._feature_coordinator.shutdown()

    def _build_batch(self, seed_nodes: list[int], seed: int) -> tuple[Data, BatchProfile]:
        t_total = time.perf_counter()

        t_s = time.perf_counter()
        sampling = gar_ml.sample_neighbors(
            self.graph_info,
            self.vertex_type,
            self.edge_type,
            seed_nodes,
            self.num_neighbors,
            seed=seed,
            chunk_manager=self._sampling_chunk_manager,
        )
        sampling_ms = (time.perf_counter() - t_s) * 1000

        n_id_list = [int(node) for node in sampling.sampled_nodes]
        src_list = [int(src_idx) for src_idx in sampling.src_indices]
        dst_list = [int(dst_idx) for dst_idx in sampling.dst_indices]

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        feature_fetch_ms = 0.0
        conversion_ms = 0.0
        if self.features:
            t_s = time.perf_counter()
            feature_table = self._feature_coordinator.submit(
                self.vertex_type,
                n_id_list,
                self.features,
            ).wait()
            feature_fetch_ms = (time.perf_counter() - t_s) * 1000

            t_s = time.perf_counter()
            x = _table_to_feature_tensor(feature_table)
            conversion_ms = (time.perf_counter() - t_s) * 1000
        else:
            x = torch.empty((len(n_id_list), 0), dtype=torch.float32)

        n_id = torch.tensor(n_id_list, dtype=torch.long)
        input_id = torch.tensor(seed_nodes, dtype=torch.long)
        num_sampled_nodes = torch.tensor(list(sampling.num_sampled_nodes_per_hop), dtype=torch.long)
        num_sampled_edges = torch.tensor(list(sampling.num_sampled_edges_per_hop), dtype=torch.long)

        batch = Data(x=x, edge_index=edge_index)
        batch.batch_size = len(seed_nodes)
        batch.n_id = n_id
        batch.input_id = input_id
        batch.num_sampled_nodes = num_sampled_nodes
        batch.num_sampled_edges = num_sampled_edges
        batch.vertex_type = self.vertex_type
        batch.edge_type = self.edge_type

        prof = BatchProfile(
            total_ms=(time.perf_counter() - t_total) * 1000,
            sampling_ms=sampling_ms,
            feature_fetch_ms=feature_fetch_ms,
            conversion_ms=conversion_ms,
        )
        return batch, prof

    def __iter__(self) -> Iterator[Data]:
        for batch, _ in self._iter_batches():
            yield batch

    def profile(self) -> Iterator[tuple[Data, BatchProfile]]:
        """Like __iter__, but yields (batch, BatchProfile) with per-batch timings for performance debugging."""
        yield from self._iter_batches()

    def _iter_batches(self) -> Iterator[tuple[Data, BatchProfile]]:
        """Yields (batch, timings) for both sequential and parallel modes."""
        jobs = ((nodes, self._sample_seed()) for nodes in self._iter_input_batches())
        if self.num_workers == 0:
            for seed_nodes, seed in jobs:
                yield self._build_batch(seed_nodes, seed)
        else:
            in_flight: deque[Future] = deque()
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for seed_nodes, seed in jobs:
                    in_flight.append(executor.submit(self._build_batch, seed_nodes, seed))
                    if len(in_flight) < self.num_workers:
                        continue
                    yield in_flight.popleft().result()
                while in_flight:
                    yield in_flight.popleft().result()
