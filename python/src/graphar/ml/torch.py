from __future__ import annotations

import struct
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data

import graphar.ml as gar_ml


def _chunked_array_to_tensor(column: pa.ChunkedArray) -> torch.Tensor:
    combined = column.combine_chunks()
    numpy_array = combined.to_numpy(zero_copy_only=False)
    # from_numpy may warn if underlying array is read-only; this loader is read-only so it's ok
    tensor = torch.from_numpy(numpy_array)
    return tensor.to(torch.float32)


def _table_to_feature_tensor(table: pa.Table) -> torch.Tensor:
    if table.num_columns == 0:
        return torch.empty((table.num_rows, 0), dtype=torch.float32)
    columns = [_chunked_array_to_tensor(table.column(i)) for i in range(table.num_columns)]
    return torch.stack(columns, dim=1)


def _properties_for_vertex(graph_info, vertex_type: str) -> list[str]:
    """Collect all properties when features=None"""
    vertex_info = graph_info.get_vertex_info(vertex_type)
    if vertex_info is None:
        msg = f"Vertex type '{vertex_type}' not found"  # TODO: separate error type or function?
        raise ValueError(msg)
    properties: list[str] = []
    for group in vertex_info.get_property_groups():
        for prop in group.get_properties():
            properties.append(prop.name)
    return properties


def _vertex_count(graph_info, vertex_type: str) -> int:
    vertex_info = graph_info.get_vertex_info(vertex_type)
    if vertex_info is None:
        msg = f"Vertex type '{vertex_type}' not found"
        raise ValueError(msg)
    relative_path = vertex_info.get_vertices_num_file_path()
    prefix = graph_info.get_prefix()
    if prefix.startswith("file://"):  # TODO: possible? need to check
        prefix = prefix.removeprefix("file://")
    path = prefix.rstrip("/") + "/" + relative_path.lstrip("/")
    with Path(path).open("rb") as f:
        data = f.read()
    if len(data) == 8:
        return int(struct.unpack("<q", data)[0])
    return int(data.decode("utf-8").strip())


# _Timer lives here rather than in exps/benchmarks/ so GARNeighborLoader can use it
# without importing library code from benchmark/experiment code (reversed dependency).
# timings.py re-exports it so benchmark modules have a single import point.
class _Timer:
    """Context manager that appends elapsed seconds to store[name]."""

    __slots__ = ("_name", "_store", "_t")

    def __init__(self, name: str, store: dict) -> None:
        self._name = name
        self._store = store

    def __enter__(self) -> _Timer:
        self._t = time.perf_counter()
        return self

    def __exit__(self, exc_type, *_) -> None:
        if exc_type is None:
            self._store[self._name].append(time.perf_counter() - self._t)


def _is_numeric_type_name(type_name: str) -> bool:
    return type_name in {  # TODO: refactor to constant
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float",
        "float16",
        "float32",
        "float64",
        "double",
        "bool",
    }


def _validate_numeric_features(graph_info, vertex_type: str, features: Sequence[str]) -> None:
    if not features:
        return
    vertex_info = graph_info.get_vertex_info(vertex_type)
    if vertex_info is None:
        msg = f"Vertex type '{vertex_type}' not found"
        raise ValueError(msg)
    invalid: list[tuple[str, str]] = []
    for name in features:
        type_name = vertex_info.get_property_type(name).to_type_name()  # TODO: add type mapping to original module?
        if not _is_numeric_type_name(type_name):
            invalid.append((name, type_name))
    if invalid:
        details = ", ".join(f"{name}:{type_name}" for name, type_name in invalid)
        msg = f"Only numeric features are supported. Got non-numeric: {details}"
        raise ValueError(msg)

# TODO: add comments and/or simplify
def _normalize_input_nodes(
    graph_info,
    vertex_type: str,
    input_nodes: Sequence[int] | torch.Tensor | None,
) -> list[int]:
    if input_nodes is None:
        return list(range(_vertex_count(graph_info, vertex_type)))
    if isinstance(input_nodes, torch.Tensor):
        if input_nodes.dtype == torch.bool:
            return input_nodes.nonzero(as_tuple=False).view(-1).tolist()
        return input_nodes.view(-1).to(dtype=torch.long).tolist()
    if len(input_nodes) == 0:
        return []
    if all(isinstance(node, bool) for node in input_nodes):
        return [idx for idx, keep in enumerate(input_nodes) if keep]
    return [int(node) for node in input_nodes]

# TODO: do we need this strange function?
def _as_unique_list(values: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    unique_values: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values

# TODO: unclear what this function does
def _hop_stats(
    num_hops: int, seed_count: int, edge_index: torch.Tensor
) -> tuple[list[int], list[int]]:
    if seed_count == 0:
        return [0] * num_hops, [0] * num_hops
    if edge_index.numel() == 0:
        return [0] * num_hops, [0] * num_hops

    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    depth_by_node = dict.fromkeys(range(seed_count), 0)
    num_sampled_edges = [0] * num_hops
    num_sampled_nodes = [0] * num_hops

    for hop in range(num_hops):
        new_nodes: set[int] = set()
        for src, dst in zip(sources, targets):
            src_depth = depth_by_node.get(src)
            if src_depth != hop:
                continue
            num_sampled_edges[hop] += 1
            if dst in depth_by_node:
                continue
            depth_by_node[dst] = hop + 1
            new_nodes.add(dst)
        num_sampled_nodes[hop] = len(new_nodes)
    return num_sampled_nodes, num_sampled_edges


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
    ) -> None:
        if batch_size <= 0:
            msg = "batch_size must be > 0"
            raise ValueError(msg)
        if not num_neighbors:
            msg = "num_neighbors must not be empty"
            raise ValueError(msg)
        self.graph_info = graph_info
        self.vertex_type = vertex_type
        self.edge_type = edge_type
        self.num_neighbors = [int(v) for v in num_neighbors]  # TODO: review whether casts are needed
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._input_nodes = _as_unique_list(
            _normalize_input_nodes(graph_info, vertex_type, input_nodes)
        )
        self.features = (
            _properties_for_vertex(graph_info, vertex_type)
            if features is None
            else [str(name) for name in features]
        )
        _validate_numeric_features(graph_info, vertex_type, self.features)
        self._rng = torch.Generator()  # used for both dataset shuffling and sampling seeds
        self._rng.manual_seed(int(torch.initial_seed()))
        self.timings: defaultdict = defaultdict(list)

    def __len__(self) -> int:
        if not self._input_nodes:
            return 0
        return (len(self._input_nodes) + self.batch_size - 1) // self.batch_size

    def _iter_input_batches(self) -> Iterator[list[int]]:
        total = len(self._input_nodes)
        if total == 0:
            return
        if self.shuffle:
            order = torch.randperm(total, generator=self._rng).tolist()
            ordered_nodes = [self._input_nodes[i] for i in order]  # TODO: does it make sense?
        else:
            ordered_nodes = self._input_nodes
        for start in range(0, total, self.batch_size):
            yield ordered_nodes[start : start + self.batch_size]

    def _sample_seed(self) -> int:
        return int(
            torch.randint(2**32, (1,), generator=self._rng, dtype=torch.int64).item()
        )

    def _build_batch(self, seed_nodes: list[int]) -> Data:
        seed = self._sample_seed()

        with _Timer("sampling", self.timings):
            sampling = gar_ml.sample_neighbors(
                self.graph_info,
                self.vertex_type,
                self.edge_type,
                seed_nodes,
                self.num_neighbors,
                seed=seed,
            )

        n_id_list = [int(node) for node in sampling.sampled_nodes]
        src_list = [int(src_idx) for src_idx in sampling.src_indices]
        dst_list = [int(dst_idx) for dst_idx in sampling.dst_indices]

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        if self.features:
            with _Timer("feature_fetch", self.timings):
                feature_table = gar_ml.get_node_features(
                    self.graph_info, self.vertex_type, n_id_list, self.features
                )
            with _Timer("conversion", self.timings):
                x = _table_to_feature_tensor(feature_table)
        else:
            x = torch.empty((len(n_id_list), 0), dtype=torch.float32)

        n_id = torch.tensor(n_id_list, dtype=torch.long)
        input_id = torch.tensor(seed_nodes, dtype=torch.long)
        num_sampled_nodes, num_sampled_edges = _hop_stats(
            len(self.num_neighbors), len(seed_nodes), edge_index
        )

        batch = Data(x=x, edge_index=edge_index)
        batch.batch_size = len(seed_nodes)
        batch.n_id = n_id
        batch.input_id = input_id
        batch.num_sampled_nodes = torch.tensor(num_sampled_nodes, dtype=torch.long)  # TODO: why list? need to check
        batch.num_sampled_edges = torch.tensor(num_sampled_edges, dtype=torch.long)  # TODO: why list? need to check
        batch.vertex_type = self.vertex_type
        batch.edge_type = self.edge_type

        return batch

    def __iter__(self) -> Iterator[Data]:
        for seed_nodes in self._iter_input_batches():
            yield self._build_batch(seed_nodes)
