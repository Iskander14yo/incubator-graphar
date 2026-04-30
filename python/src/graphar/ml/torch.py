from __future__ import annotations

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
    total_started_at: float
    sampling_ms: float
    feature_handle: object | None


def _chunked_array_to_tensor(column: pa.ChunkedArray, name: str) -> torch.Tensor:
    combined = column.combine_chunks()
    numpy_array = combined.to_numpy(zero_copy_only=False)
    try:
        tensor = torch.from_numpy(numpy_array)
    except TypeError:
        msg = f"Feature '{name}' cannot be converted to a tensor (non-numeric type: {column.type})"
        raise TypeError(msg) from None
    return tensor.to(torch.float32)


def _table_to_feature_tensor(table: pa.Table) -> torch.Tensor:
    """Convert pyarrow Table (from cpp binding) to pytorch Tensor."""
    if table.num_columns == 0:
        return torch.empty((table.num_rows, 0), dtype=torch.float32)
    columns = [_chunked_array_to_tensor(table.column(i), table.schema.field(i).name) for i in range(table.num_columns)]
    return torch.stack(columns, dim=1)


def _properties_for_vertex(graph_info, vertex_type: str) -> list[str]:
    """Collect all properties when features=None."""
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
    """For 3 input shapes: None / bool-mask / int-list."""
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


def _feature_pipeline_stats_to_dict(stats) -> dict[str, int]:
    return {
        "submitted_batches": int(stats.submitted_batches),
        "completed_batches": int(stats.completed_batches),
        "pending_batches_peak": int(stats.pending_batches_peak),
        "active_chunk_keys_peak": int(stats.active_chunk_keys_peak),
        "chunk_subscriptions": int(stats.chunk_subscriptions),
        "chunk_reads": int(stats.chunk_reads),
        "chunk_reuses": int(stats.chunk_reuses),
        "stitch_tasks": int(stats.stitch_tasks),
        "stitch_wait_ms_sum": int(stats.stitch_wait_ms_sum),
        "stitch_service_ms_sum": int(stats.stitch_service_ms_sum),
    }


def _empty_feature_pipeline_stats() -> dict[str, int]:
    return {
        "submitted_batches": 0,
        "completed_batches": 0,
        "pending_batches_peak": 0,
        "active_chunk_keys_peak": 0,
        "chunk_subscriptions": 0,
        "chunk_reads": 0,
        "chunk_reuses": 0,
        "stitch_tasks": 0,
        "stitch_wait_ms_sum": 0,
        "stitch_service_ms_sum": 0,
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
        num_samplers: int = 0,
        prefetch_batches: int = 0,
        edge_ram_for_loader_mb: int = 0,
        feature_ram_for_loader_mb: int = 0,
        num_readers: int | None = None,
        num_stitchers: int = 1,
        feature_cursor_count: int | None = None,
        feature_cursor_trail_chunks: int = 10,
    ) -> None:
        if batch_size <= 0:
            msg = "batch_size must be > 0"
            raise ValueError(msg)
        if not num_neighbors:
            msg = "num_neighbors must not be empty"
            raise ValueError(msg)
        if num_samplers < 0:
            msg = "num_samplers must be >= 0"
            raise ValueError(msg)
        if prefetch_batches < 0:
            msg = "prefetch_batches must be >= 0"
            raise ValueError(msg)
        if edge_ram_for_loader_mb < 0:
            msg = "edge_ram_for_loader_mb must be >= 0"
            raise ValueError(msg)
        if feature_ram_for_loader_mb < 0:
            msg = "feature_ram_for_loader_mb must be >= 0"
            raise ValueError(msg)
        if num_readers is None:
            num_readers = 1 if feature_cursor_count is None else feature_cursor_count
        if feature_cursor_count is not None and num_readers != feature_cursor_count:
            msg = "num_readers must match feature_cursor_count when both are provided"
            raise ValueError(msg)
        if num_readers <= 0:
            msg = "num_readers must be > 0"
            raise ValueError(msg)
        if num_stitchers <= 0:
            msg = "num_stitchers must be > 0"
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
        self.num_samplers = num_samplers
        self.prefetch_batches = prefetch_batches
        self.num_readers = int(num_readers)
        self.num_stitchers = int(num_stitchers)
        edge_chunk_manager_options = gar_ml._ChunkReadManagerOptions()
        edge_chunk_manager_options.ram_budget_bytes = int(edge_ram_for_loader_mb) * 1024 * 1024
        self._sampling_chunk_manager = gar_ml._ChunkReadManager(edge_chunk_manager_options)
        feature_chunk_manager_options = gar_ml._ChunkReadManagerOptions()
        feature_chunk_manager_options.ram_budget_bytes = int(feature_ram_for_loader_mb) * 1024 * 1024
        feature_chunk_manager_options.feature_cursor_count = self.num_readers
        feature_chunk_manager_options.feature_cursor_trail_capacity_chunks = int(feature_cursor_trail_chunks)
        self._feature_chunk_manager = gar_ml._ChunkReadManager(feature_chunk_manager_options)
        if input_nodes is None and not shuffle:
            self._input_nodes = None
            self._input_node_count = graph_info.get_vertex_count(vertex_type)
        else:
            self._input_nodes = list(
                dict.fromkeys(_normalize_input_nodes(graph_info, vertex_type, input_nodes))
            )
            self._input_node_count = len(self._input_nodes)
        self.features = _properties_for_vertex(graph_info, vertex_type) if features is None else features
        self._rng = torch.Generator()
        self._rng.manual_seed(int(torch.initial_seed()))
        self._feature_pipeline = None
        if self.features:
            feature_pipeline_options = gar_ml._FeaturePipelineOptions()
            feature_pipeline_options.num_readers = self.num_readers
            feature_pipeline_options.num_stitchers = self.num_stitchers
            feature_pipeline_options.max_active_batches = self._max_pending_batches()
            feature_pipeline_options.max_queued_stitch_tasks = (
                feature_pipeline_options.max_active_batches * self.num_stitchers
            )
            self._feature_pipeline = gar_ml._FeaturePipelineCoordinator(
                self._feature_chunk_manager,
                feature_pipeline_options,
            )

    def __len__(self) -> int:
        if self._input_node_count == 0:
            return 0
        return (self._input_node_count + self.batch_size - 1) // self.batch_size

    def _max_pending_batches(self) -> int:
        if self.prefetch_batches > 0:
            return self.prefetch_batches
        if self.num_samplers > 0:
            return self.num_samplers
        return 1

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
        return int(torch.randint(2**32, (1,), generator=self._rng, dtype=torch.int64).item())

    def chunk_manager_stats(self) -> dict[str, int]:
        return self.sampling_chunk_manager_stats()

    def sampling_chunk_manager_stats(self) -> dict[str, int]:
        return _chunk_read_stats_to_dict(self._sampling_chunk_manager.stats())

    def feature_chunk_manager_stats(self) -> dict[str, int]:
        return _chunk_read_stats_to_dict(self._feature_chunk_manager.stats())

    def feature_pipeline_stats(self) -> dict[str, int]:
        if self._feature_pipeline is None:
            return _empty_feature_pipeline_stats()
        return _feature_pipeline_stats_to_dict(self._feature_pipeline.stats())

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

    def close(self) -> None:
        if self._feature_pipeline is not None:
            self._feature_pipeline.shutdown()
        self._feature_chunk_manager.shutdown()

    def _sample_and_submit_batch(self, seed_nodes: list[int], seed: int) -> _SampledBatch:
        total_started_at = time.perf_counter()
        t_s = total_started_at
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

        sampled_nodes = [int(node) for node in sampling.sampled_nodes]
        src_indices = [int(src_idx) for src_idx in sampling.src_indices]
        dst_indices = [int(dst_idx) for dst_idx in sampling.dst_indices]
        feature_handle = None
        if self.features:
            assert self._feature_pipeline is not None
            feature_handle = self._feature_pipeline.submit_sampled_batch(
                self.graph_info,
                self.vertex_type,
                sampled_nodes,
                self.features,
            )
        return _SampledBatch(
            seed_nodes=list(seed_nodes),
            sampled_nodes=sampled_nodes,
            src_indices=src_indices,
            dst_indices=dst_indices,
            num_sampled_nodes=[int(value) for value in sampling.num_sampled_nodes_per_hop],
            num_sampled_edges=[int(value) for value in sampling.num_sampled_edges_per_hop],
            total_started_at=total_started_at,
            sampling_ms=sampling_ms,
            feature_handle=feature_handle,
        )

    def _materialize_batch(self, sampled: _SampledBatch) -> tuple[Data, BatchProfile]:
        if sampled.src_indices:
            edge_index = torch.tensor([sampled.src_indices, sampled.dst_indices], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        feature_fetch_ms = 0.0
        conversion_ms = 0.0
        if self.features:
            assert sampled.feature_handle is not None
            feature_table = sampled.feature_handle.wait()
            feature_fetch_ms = float(sampled.feature_handle.feature_fetch_ms())

            t_s = time.perf_counter()
            x = _table_to_feature_tensor(feature_table)
            conversion_ms = (time.perf_counter() - t_s) * 1000
        else:
            x = torch.empty((len(sampled.sampled_nodes), 0), dtype=torch.float32)

        n_id = torch.tensor(sampled.sampled_nodes, dtype=torch.long)
        input_id = torch.tensor(sampled.seed_nodes, dtype=torch.long)
        num_sampled_nodes = torch.tensor(sampled.num_sampled_nodes, dtype=torch.long)
        num_sampled_edges = torch.tensor(sampled.num_sampled_edges, dtype=torch.long)

        batch = Data(x=x, edge_index=edge_index)
        batch.batch_size = len(sampled.seed_nodes)
        batch.n_id = n_id
        batch.input_id = input_id
        batch.num_sampled_nodes = num_sampled_nodes
        batch.num_sampled_edges = num_sampled_edges
        batch.vertex_type = self.vertex_type
        batch.edge_type = self.edge_type

        prof = BatchProfile(
            total_ms=(time.perf_counter() - sampled.total_started_at) * 1000,
            sampling_ms=sampled.sampling_ms,
            feature_fetch_ms=feature_fetch_ms,
            conversion_ms=conversion_ms,
        )
        return batch, prof

    def _build_batch(self, seed_nodes: list[int], seed: int) -> tuple[Data, BatchProfile]:
        return self._materialize_batch(self._sample_and_submit_batch(seed_nodes, seed))

    def __iter__(self) -> Iterator[Data]:
        for batch, _ in self._iter_batches():
            yield batch

    def profile(self) -> Iterator[tuple[Data, BatchProfile]]:
        """Like __iter__, but yields (batch, BatchProfile)."""
        yield from self._iter_batches()

    def _iter_batches(self) -> Iterator[tuple[Data, BatchProfile]]:
        jobs = ((nodes, self._sample_seed()) for nodes in self._iter_input_batches())
        max_pending = self._max_pending_batches()

        if self.num_samplers == 0:
            next_seq = 0
            next_to_yield = 0
            jobs_exhausted = False
            completed: dict[int, _SampledBatch] = {}

            def submit_until_full() -> None:
                nonlocal next_seq, jobs_exhausted
                while not jobs_exhausted and len(completed) < max_pending:
                    try:
                        seed_nodes, seed = next(jobs)
                    except StopIteration:
                        jobs_exhausted = True
                        return
                    completed[next_seq] = self._sample_and_submit_batch(seed_nodes, seed)
                    next_seq += 1

            submit_until_full()
            while completed:
                yield self._materialize_batch(completed.pop(next_to_yield))
                next_to_yield += 1
                submit_until_full()
            return

        next_seq = 0
        next_to_yield = 0
        jobs_exhausted = False
        completed: dict[int, _SampledBatch] = {}
        in_flight: dict[Future, int] = {}

        def submit_until_full(executor: ThreadPoolExecutor) -> None:
            nonlocal next_seq, jobs_exhausted
            while not jobs_exhausted and len(in_flight) + len(completed) < max_pending:
                try:
                    seed_nodes, seed = next(jobs)
                except StopIteration:
                    jobs_exhausted = True
                    return
                future = executor.submit(self._sample_and_submit_batch, seed_nodes, seed)
                in_flight[future] = next_seq
                next_seq += 1

        with ThreadPoolExecutor(max_workers=self.num_samplers) as executor:
            submit_until_full(executor)
            while in_flight or completed:
                while next_to_yield in completed:
                    yield self._materialize_batch(completed.pop(next_to_yield))
                    next_to_yield += 1
                    submit_until_full(executor)
                if not in_flight:
                    continue
                done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    completed[in_flight.pop(future)] = future.result()
                submit_until_full(executor)
