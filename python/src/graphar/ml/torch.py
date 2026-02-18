from __future__ import annotations

from collections.abc import Iterator, Sequence

import pyarrow as pa
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data

from . import get_node_features, sample_neighbors


def _is_numeric_arrow_type(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_boolean(data_type)
    )


def _chunked_array_to_tensor(column: pa.ChunkedArray) -> torch.Tensor:
    combined = column.combine_chunks()
    if not _is_numeric_arrow_type(combined.type):
        raise TypeError(f"Feature column '{column}' has non-numeric type: {combined.type}")
    numpy_array = combined.to_numpy(zero_copy_only=False)
    tensor = torch.from_numpy(numpy_array)
    if tensor.dtype == torch.bool:
        return tensor.to(torch.float32)
    if tensor.is_floating_point():
        return tensor.to(torch.float32)
    return tensor.to(torch.float32)


def _table_to_feature_tensor(table: pa.Table) -> torch.Tensor:
    if table.num_columns == 0:
        return torch.empty((table.num_rows, 0), dtype=torch.float32)
    columns = [_chunked_array_to_tensor(table.column(i)) for i in range(table.num_columns)]
    return torch.stack(columns, dim=1)


def _properties_for_vertex(graph_info, vertex_type: str) -> list[str]:
    vertex_info = graph_info.get_vertex_info(vertex_type)
    if vertex_info is None:
        raise ValueError(f"Vertex type '{vertex_type}' not found")
    properties: list[str] = []
    for group in vertex_info.get_property_groups():
        for prop in group.get_properties():
            properties.append(prop.name)
    return properties


def _vertex_count(graph_info, vertex_type: str) -> int:
    vertex_info = graph_info.get_vertex_info(vertex_type)
    if vertex_info is None:
        raise ValueError(f"Vertex type '{vertex_type}' not found")
    relative_path = vertex_info.get_vertices_num_file_path()
    prefix = graph_info.get_prefix()
    if prefix.startswith("file://"):
        prefix = prefix.removeprefix("file://")
    path = prefix.rstrip("/") + "/" + relative_path.lstrip("/")
    with open(path, encoding="utf-8") as f:
        return int(f.read().strip())


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


def _as_unique_list(values: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    unique_values: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values

# todo: проверить этот инвариант в сорсе
def _reorder_sampled_nodes(sampled_nodes: Sequence[int], seed_nodes: Sequence[int]) -> list[int]:
    seed_set = set(seed_nodes)
    rest = [node for node in sampled_nodes if node not in seed_set]
    return [*seed_nodes, *rest]


def _hop_stats(
    num_hops: int, seed_count: int, edge_index: torch.Tensor
) -> tuple[list[int], list[int]]:
    if seed_count == 0:
        return [0] * num_hops, [0] * num_hops
    if edge_index.numel() == 0:
        return [0] * num_hops, [0] * num_hops

    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    depth_by_node = {idx: 0 for idx in range(seed_count)}
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
            raise ValueError("batch_size must be > 0")
        if not num_neighbors:
            raise ValueError("num_neighbors must not be empty")
        self.graph_info = graph_info
        self.vertex_type = vertex_type
        self.edge_type = edge_type
        self.num_neighbors = [int(v) for v in num_neighbors]  # todo: убрать все эти конвертации
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
        self._epoch = 0  # todo: подумать на тему эпохи в качестве сида

    def __len__(self) -> int:
        if not self._input_nodes:
            return 0
        return (len(self._input_nodes) + self.batch_size - 1) // self.batch_size

    def _iter_input_batches(self) -> Iterator[list[int]]:
        total = len(self._input_nodes)
        if total == 0:
            return
        if self.shuffle:
            order = torch.randperm(total).tolist()
            ordered_nodes = [self._input_nodes[i] for i in order]
        else:
            ordered_nodes = self._input_nodes
        for start in range(0, total, self.batch_size):
            yield ordered_nodes[start : start + self.batch_size]

    def _sample_seed(self) -> int:
        if self.shuffle:
            return int(torch.initial_seed())
        return self._epoch

    def _build_batch(self, seed_nodes: list[int]) -> Data:
        seed = self._sample_seed()
        sampling = sample_neighbors(
            self.graph_info,
            self.vertex_type,
            self.edge_type,
            seed_nodes,
            self.num_neighbors,
            seed=seed,
        )

        sampled_nodes = [int(node) for node in sampling.sampled_nodes]
        n_id_list = _reorder_sampled_nodes(sampled_nodes, seed_nodes)

        old_nodes = [int(node) for node in sampling.sampled_nodes]
        old_idx_to_node = old_nodes
        new_idx_by_node = {node: idx for idx, node in enumerate(n_id_list)}

        src_list: list[int] = []
        dst_list: list[int] = []
        for src_idx, dst_idx in zip(sampling.src_indices, sampling.dst_indices):
            src_node = old_idx_to_node[int(src_idx)]
            dst_node = old_idx_to_node[int(dst_idx)]
            src_list.append(new_idx_by_node[src_node])
            dst_list.append(new_idx_by_node[dst_node])

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        if self.features:
            feature_table = get_node_features(
                self.graph_info, self.vertex_type, n_id_list, self.features
            )
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
        batch.num_sampled_nodes = torch.tensor(num_sampled_nodes, dtype=torch.long)
        batch.num_sampled_edges = torch.tensor(num_sampled_edges, dtype=torch.long)
        batch.vertex_type = self.vertex_type
        batch.edge_type = self.edge_type
        return batch

    def __iter__(self) -> Iterator[Data]:
        try:
            for seed_nodes in self._iter_input_batches():
                yield self._build_batch(seed_nodes)
        finally:
            if not self.shuffle:
                self._epoch += 1
