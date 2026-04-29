from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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


@dataclass
class _SampledBatch:
    seed_nodes: list[int]
    sampled_nodes: list[int]
    src_indices: list[int]
    dst_indices: list[int]
    num_sampled_nodes: list[int]
    num_sampled_edges: list[int]
    sampling_ms: float


@dataclass
class _FeatureWindowMetrics:
    windows: int = 0
    window_batches_sum: int = 0
    window_batches_max: int = 0
    window_fetch_ms_sum: float = 0.0
    window_conversion_ms_sum: float = 0.0
    window_nodes_sum: int = 0
    window_unique_nodes_sum: int = 0

    def record(
        self,
        window_batches: int,
        window_fetch_ms: float,
        window_conversion_ms: float,
        window_nodes: int,
        window_unique_nodes: int,
    ) -> None:
        self.windows += 1
        self.window_batches_sum += window_batches
        self.window_batches_max = max(self.window_batches_max, window_batches)
        self.window_fetch_ms_sum += window_fetch_ms
        self.window_conversion_ms_sum += window_conversion_ms
        self.window_nodes_sum += window_nodes
        self.window_unique_nodes_sum += window_unique_nodes

    def to_dict(self) -> dict[str, float | int]:
        if self.windows == 0:
            return {
                "windows": 0,
                "window_batches_sum": 0,
                "window_batches_mean": 0.0,
                "window_batches_max": 0,
                "window_fetch_ms_sum": 0.0,
                "window_fetch_ms_mean": 0.0,
                "window_conversion_ms_sum": 0.0,
                "window_conversion_ms_mean": 0.0,
                "window_nodes_sum": 0,
                "window_unique_nodes_sum": 0,
                "window_reuse_ratio": 0.0,
            }
        return {
            "windows": self.windows,
            "window_batches_sum": self.window_batches_sum,
            "window_batches_mean": self.window_batches_sum / self.windows,
            "window_batches_max": self.window_batches_max,
            "window_fetch_ms_sum": self.window_fetch_ms_sum,
            "window_fetch_ms_mean": self.window_fetch_ms_sum / self.windows,
            "window_conversion_ms_sum": self.window_conversion_ms_sum,
            "window_conversion_ms_mean": self.window_conversion_ms_sum / self.windows,
            "window_nodes_sum": self.window_nodes_sum,
            "window_unique_nodes_sum": self.window_unique_nodes_sum,
            "window_reuse_ratio": (
                self.window_nodes_sum / self.window_unique_nodes_sum
                if self.window_unique_nodes_sum
                else 0.0
            ),
        }


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
        prefetch_windows: int = 0,
        feature_buffer_batches: int = 1,
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
        if prefetch_windows < 0:
            msg = "prefetch_windows must be >= 0"
            raise ValueError(msg)
        if feature_buffer_batches <= 0:
            msg = "feature_buffer_batches must be > 0"
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
        self.prefetch_windows = prefetch_windows
        self.feature_buffer_batches = feature_buffer_batches
        edge_chunk_manager_options = gar_ml._ChunkReadManagerOptions()
        edge_chunk_manager_options.ram_budget_bytes = int(edge_ram_for_loader_mb) * 1024 * 1024
        self._sampling_chunk_manager = gar_ml._ChunkReadManager(edge_chunk_manager_options)
        feature_chunk_manager_options = gar_ml._ChunkReadManagerOptions()
        feature_chunk_manager_options.ram_budget_bytes = int(feature_ram_for_loader_mb) * 1024 * 1024
        feature_chunk_manager_options.feature_cursor_count = int(feature_cursor_count)
        feature_chunk_manager_options.feature_cursor_trail_capacity_chunks = int(feature_cursor_trail_chunks)
        self._feature_chunk_manager = gar_ml._ChunkReadManager(feature_chunk_manager_options)
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
        self._feature_window_metrics = _FeatureWindowMetrics()
        self._feature_window_metrics_lock = threading.Lock()

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
        stats = self._feature_chunk_manager.feature_cursor_stats()
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

    def feature_window_stats(self) -> dict[str, float | int]:
        with self._feature_window_metrics_lock:
            return self._feature_window_metrics.to_dict()

    def close(self) -> None:
        self._feature_chunk_manager.shutdown()

    def _sample_batch(self, seed_nodes: list[int], seed: int) -> _SampledBatch:
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

        return _SampledBatch(
            seed_nodes=list(seed_nodes),
            sampled_nodes=[int(node) for node in sampling.sampled_nodes],
            src_indices=[int(src_idx) for src_idx in sampling.src_indices],
            dst_indices=[int(dst_idx) for dst_idx in sampling.dst_indices],
            num_sampled_nodes=list(sampling.num_sampled_nodes_per_hop),
            num_sampled_edges=list(sampling.num_sampled_edges_per_hop),
            sampling_ms=sampling_ms,
        )

    def _materialize_window(
        self,
        sampled_batches: list[_SampledBatch],
    ) -> list[tuple[Data, BatchProfile]]:
        if not sampled_batches:
            return []
        union_nodes: list[int] = []
        row_index_by_node: dict[int, int] = {}
        window_nodes = 0
        for sampled_batch in sampled_batches:
            window_nodes += len(sampled_batch.sampled_nodes)
            for node in sampled_batch.sampled_nodes:
                if node not in row_index_by_node:
                    row_index_by_node[node] = len(union_nodes)
                    union_nodes.append(node)

        window_fetch_ms = 0.0
        window_conversion_ms = 0.0
        if self.features:
            t_s = time.perf_counter()
            feature_table = gar_ml.get_node_features(
                self.graph_info,
                self.vertex_type,
                union_nodes,
                self.features,
                chunk_manager=self._feature_chunk_manager,
            )
            window_fetch_ms = (time.perf_counter() - t_s) * 1000

            t_s = time.perf_counter()
            union_x = _table_to_feature_tensor(feature_table)
            window_conversion_ms = (time.perf_counter() - t_s) * 1000
        else:
            union_x = torch.empty((len(union_nodes), 0), dtype=torch.float32)

        window_batches = len(sampled_batches)
        feature_fetch_ms = window_fetch_ms / window_batches
        conversion_ms = window_conversion_ms / window_batches
        with self._feature_window_metrics_lock:
            self._feature_window_metrics.record(
                window_batches,
                window_fetch_ms,
                window_conversion_ms,
                window_nodes,
                len(union_nodes),
            )

        results: list[tuple[Data, BatchProfile]] = []
        for sampled_batch in sampled_batches:
            row_indices = torch.tensor(
                [row_index_by_node[node] for node in sampled_batch.sampled_nodes],
                dtype=torch.long,
            )
            x = union_x.index_select(0, row_indices)
            if sampled_batch.src_indices:
                edge_index = torch.tensor(
                    [sampled_batch.src_indices, sampled_batch.dst_indices],
                    dtype=torch.long,
                )
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            batch = Data(x=x, edge_index=edge_index)
            batch.batch_size = len(sampled_batch.seed_nodes)
            batch.n_id = torch.tensor(sampled_batch.sampled_nodes, dtype=torch.long)
            batch.input_id = torch.tensor(sampled_batch.seed_nodes, dtype=torch.long)
            batch.num_sampled_nodes = torch.tensor(sampled_batch.num_sampled_nodes, dtype=torch.long)
            batch.num_sampled_edges = torch.tensor(sampled_batch.num_sampled_edges, dtype=torch.long)
            batch.vertex_type = self.vertex_type
            batch.edge_type = self.edge_type

            prof = BatchProfile(
                total_ms=sampled_batch.sampling_ms + feature_fetch_ms + conversion_ms,
                sampling_ms=sampled_batch.sampling_ms,
                feature_fetch_ms=feature_fetch_ms,
                conversion_ms=conversion_ms,
            )
            results.append((batch, prof))
        return results

    def _build_window(
        self,
        window_jobs: list[tuple[list[int], int]],
    ) -> list[tuple[Data, BatchProfile]]:
        sampled_batches = [
            self._sample_batch(seed_nodes, seed)
            for seed_nodes, seed in window_jobs
        ]
        return self._materialize_window(sampled_batches)

    def _iter_input_windows(self) -> Iterator[list[tuple[list[int], int]]]:
        window: list[tuple[list[int], int]] = []
        for seed_nodes in self._iter_input_batches():
            window.append((seed_nodes, self._sample_seed()))
            if len(window) == self.feature_buffer_batches:
                yield window
                window = []
        if window:
            yield window

    def __iter__(self) -> Iterator[Data]:
        for batch, _ in self._iter_batches():
            yield batch

    def profile(self) -> Iterator[tuple[Data, BatchProfile]]:
        """Like __iter__, but yields (batch, BatchProfile) with per-batch timings for performance debugging."""
        yield from self._iter_batches()

    def _iter_batches(self) -> Iterator[tuple[Data, BatchProfile]]:
        """Yields (batch, timings) for both sequential and parallel modes."""
        with self._feature_window_metrics_lock:
            self._feature_window_metrics = _FeatureWindowMetrics()
        jobs = self._iter_input_windows()
        if self.num_workers == 0:
            for window_jobs in jobs:
                yield from self._build_window(window_jobs)
        else:
            max_pending = (
                self.prefetch_windows
                if self.prefetch_windows > 0
                else self.num_workers
            )
            next_seq = 0
            next_to_yield = 0
            jobs_exhausted = False
            # Finished windows can arrive out of order; keep them here until
            # the next ordered result is ready to yield.
            completed: dict[int, list[tuple[Data, BatchProfile]]] = {}
            in_flight: dict[Future, int] = {}

            def submit_until_full(executor: ThreadPoolExecutor) -> None:
                nonlocal next_seq, jobs_exhausted
                # Bound submitted-but-not-yielded work to avoid unbounded window
                # buffering while still letting workers run past a slow head.
                while not jobs_exhausted and len(in_flight) + len(completed) < max_pending:
                    try:
                        window_jobs = next(jobs)
                    except StopIteration:
                        jobs_exhausted = True
                        return
                    future = executor.submit(self._build_window, window_jobs)
                    in_flight[future] = next_seq
                    next_seq += 1

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                submit_until_full(executor)
                while in_flight or completed:
                    # Drain any newly available ordered prefix before waiting
                    # for more work to finish.
                    while next_to_yield in completed:
                        for batch_and_profile in completed.pop(next_to_yield):
                            yield batch_and_profile
                        next_to_yield += 1
                        submit_until_full(executor)
                    if not in_flight:
                        continue
                    # Wait for whichever window finishes first, then refill the
                    # submission window.
                    done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                    for future in done:
                        completed[in_flight.pop(future)] = future.result()
                    submit_until_full(executor)
