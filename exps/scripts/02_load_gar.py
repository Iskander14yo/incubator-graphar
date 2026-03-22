#!/usr/bin/env python3

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ogb.nodeproppred import NodePropPredDataset

from graphar._core import do_import
from graphar.importer.config import (
    AdjList as IAdjList,
    Edge as IEdge,
    GraphArConfig,
    ImportConfig,
    ImportSchema,
    Property as IProp,
    PropertyGroup as IPropGroup,
    Source,
    Vertex as IVertex,
)
from graphar.importer.importer import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OGB dataset into GAR format.")
    parser.add_argument("--dataset", default="ogbn-products", help="OGB dataset name.")
    parser.add_argument("--ogb-root", default="exps/datasets/ogb", help="Path to OGB cache root.")
    parser.add_argument("--output-root", default="exps/datasets/gar", help="Path to GAR output root.")
    parser.add_argument(
        "--tmp-root",
        default="exps/datasets/tmp/gar_import",
        help="Path to temporary import artifacts.",
    )
    parser.add_argument("--vertex-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--edge-chunk-size", type=int, default=10_000_000)
    parser.add_argument(
        "--vertex-write-batch-size",
        type=int,
        default=500_000,
        help="Rows per parquet write batch for vertex staging file.",
    )
    parser.add_argument(
        "--edge-write-batch-size",
        type=int,
        default=5_000_000,
        help="Rows per parquet write batch for edge table.",
    )
    return parser.parse_args()


def _graph_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / f"{dataset}.graph.yml"


def _compute_vertex_groups(feat_dim: int, max_dir_name_len: int = 240) -> list[list[str]]:
    """Return column-name groups for vertex property splitting.

    Keeps each group's concatenated dir-name within max_dir_name_len bytes.
    Groups: [id], feature batches, [label].
    """
    groups: list[list[str]] = [["id"]]
    current: list[str] = []
    current_len = 0
    for i in range(feat_dim):
        name = f"f{i:03d}"
        added = len(name) + (1 if current else 0)
        if current and current_len + added > max_dir_name_len:
            groups.append(current)
            current = []
            current_len = 0
            added = len(name)
        current.append(name)
        current_len += added
    if current:
        groups.append(current)
    groups.append(["label"])
    return groups


def _write_vertex_parquet_flat(
    path: Path,
    node_feat: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> None:
    """Write vertex columns to a flat Parquet for do_import.

    Includes a '_dst_id' column (== 'id') as a workaround for a do_import bug
    where src_prop == dst_prop in homogeneous graphs causes both edge endpoints
    to resolve to the same source column. See python/src/bindings/importer.h.
    """
    num_nodes = node_feat.shape[0]
    feat_dim = node_feat.shape[1]
    labels_1d = labels.reshape(-1)

    col_names = ["id"] + [f"f{i:03d}" for i in range(feat_dim)] + ["label", "_dst_id"]
    schema = pa.schema(
        [("id", pa.int64())]
        + [(f"f{i:03d}", pa.float32()) for i in range(feat_dim)]
        + [("label", pa.int64()), ("_dst_id", pa.int64())]
    )

    if path.exists():
        path.unlink()

    writer = pq.ParquetWriter(path, schema=schema)
    try:
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            idx = np.arange(start, end, dtype=np.int64)
            feat_T = np.ascontiguousarray(node_feat[start:end].T)
            arrays = (
                [pa.array(idx)]
                + [pa.array(feat_T[i]) for i in range(feat_dim)]
                + [
                    pa.array(labels_1d[start:end].astype(np.int64, copy=False)),
                    pa.array(idx),  # _dst_id == id; workaround, see above
                ]
            )
            writer.write_table(pa.table(dict(zip(col_names, arrays)), schema=schema))
    finally:
        writer.close()


def _write_edges_parquet_chunked(path: Path, edge_index: np.ndarray, batch_size: int) -> None:
    total_edges = edge_index.shape[1]
    schema = pa.schema([("src_id", pa.int64()), ("dst_id", pa.int64())])
    if path.exists():
        path.unlink()

    writer = pq.ParquetWriter(path, schema=schema)
    try:
        for start in range(0, total_edges, batch_size):
            end = min(start + batch_size, total_edges)
            src = pa.array(edge_index[0, start:end].astype(np.int64, copy=False), type=pa.int64())
            dst = pa.array(edge_index[1, start:end].astype(np.int64, copy=False), type=pa.int64())
            writer.write_table(pa.Table.from_arrays([src, dst], schema=schema))
    finally:
        writer.close()


def _build_import_config(
    output_dir: Path,
    dataset: str,
    vertex_groups: list[list[str]],
    vertex_chunk_size: int,
    edge_chunk_size: int,
    vertices_parquet: Path,
    edges_parquet: Path,
) -> ImportConfig:
    prop_groups: list[IPropGroup] = []
    for group in vertex_groups:
        props = []
        for name in group:
            if name == "id":
                props.append(IProp(name="id", data_type="int64", is_primary=True))
            elif name == "label":
                props.append(IProp(name="label", data_type="int64"))
            else:
                props.append(IProp(name=name, data_type="float"))
        prop_groups.append(IPropGroup(properties=props))

    # Workaround for do_import bug: in homogeneous graphs src_prop == dst_prop == "id"
    # causes reversed_columns map to be overwritten so both endpoints resolve to the same
    # column after renaming. We use a duplicate "_dst_id" property (== "id") so that
    # src_prop="id" and dst_prop="_dst_id" are distinct. See importer.h ~L421-L487.
    # Fix: track original column names before renaming; drop this group + the column
    # in _write_vertex_parquet_flat, and set dst_prop="id" in the edge config.
    prop_groups.append(IPropGroup(properties=[IProp(name="_dst_id", data_type="int64")]))

    all_prop_names = [p.name for pg in prop_groups for p in pg.properties]
    vertex_source_columns = {name: name for name in all_prop_names}

    vertex = IVertex(
        type="node",
        chunk_size=vertex_chunk_size,
        validate_level="no",
        property_groups=prop_groups,
        sources=[Source(path=str(vertices_parquet.resolve()), columns=vertex_source_columns)],
    )

    edge = IEdge(
        edge_type="edge",
        src_type="node",
        src_prop="id",
        dst_type="node",
        dst_prop="_dst_id",  # workaround; see above
        chunk_size=edge_chunk_size,
        validate_level="no",
        adj_lists=[
            IAdjList(ordered=True, aligned_by="src"),
            IAdjList(ordered=True, aligned_by="dst"),
        ],
        property_groups=[],
        sources=[Source(
            path=str(edges_parquet.resolve()),
            columns={"src_id": "id", "dst_id": "_dst_id"},
        )],
    )

    return ImportConfig(
        graphar=GraphArConfig(
            path=str(output_dir.resolve()),  # Arrow filesystem requires absolute path
            name=dataset,
            vertex_chunk_size=vertex_chunk_size,
            edge_chunk_size=edge_chunk_size,
        ),
        import_schema=ImportSchema(vertices=[vertex], edges=[edge]),
    )


def _write_graph_yml(output_dir: Path, dataset: str) -> None:
    # yaml.dump emits list items at column 0 ("- item"), but yaml-cpp requires
    # block sequences to be indented relative to their parent key ("  - item").
    prefix = str(output_dir.resolve()) + "/"
    content = (
        f"name: {dataset}\n"
        f"prefix: {prefix}\n"
        f"vertices:\n"
        f"  - node.vertex.yml\n"
        f"edges:\n"
        f"  - node_edge_node.edge.yml\n"
        f"version: gar/v1\n"
    )
    (output_dir / f"{dataset}.graph.yml").write_text(content)


def _load_ogb_graph(dataset: str, root: str) -> tuple[dict, np.ndarray]:
    try:
        ogb = NodePropPredDataset(name=dataset, root=root)
        graph, labels_raw = ogb[0]
        return graph, np.asarray(labels_raw)
    except zipfile.BadZipFile:
        root_path = Path(root)
        for zip_path in root_path.rglob("*.zip"):
            zip_path.unlink(missing_ok=True)
        ogb = NodePropPredDataset(name=dataset, root=root)
        graph, labels_raw = ogb[0]
        return graph, np.asarray(labels_raw)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_root) / args.dataset
    graph_yml = _graph_path(output_dir, args.dataset)

    graph, labels = _load_ogb_graph(args.dataset, args.ogb_root)
    node_feat = graph["node_feat"]
    edge_index = graph["edge_index"]

    if node_feat.ndim != 2:
        msg = f"Expected node_feat to be 2D, got shape={node_feat.shape}"
        raise ValueError(msg)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        msg = f"Expected edge_index shape (2, E), got shape={edge_index.shape}"
        raise ValueError(msg)

    vertex_groups = _compute_vertex_groups(node_feat.shape[1])

    tmp_dir = Path(args.tmp_root) / args.dataset
    tmp_dir.mkdir(parents=True, exist_ok=True)
    vertices_parquet = tmp_dir / "vertices.parquet"
    edges_parquet = tmp_dir / "edges.parquet"

    print("Writing vertex parquet for import...")
    _write_vertex_parquet_flat(
        path=vertices_parquet,
        node_feat=node_feat,
        labels=labels,
        batch_size=args.vertex_write_batch_size,
    )

    print("Writing edge parquet for import...")
    _write_edges_parquet_chunked(
        path=edges_parquet,
        edge_index=edge_index,
        batch_size=args.edge_write_batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building import config...")
    import_config = _build_import_config(
        output_dir=output_dir,
        dataset=args.dataset,
        vertex_groups=vertex_groups,
        vertex_chunk_size=args.vertex_chunk_size,
        edge_chunk_size=args.edge_chunk_size,
        vertices_parquet=vertices_parquet,
        edges_parquet=edges_parquet,
    )
    validate(import_config)

    print("Importing to GAR via C++ importer...")
    do_import(import_config.model_dump())

    _write_graph_yml(output_dir, args.dataset)

    if not graph_yml.exists():
        msg = f"GAR conversion completed but graph file not found: {graph_yml}"
        raise FileNotFoundError(msg)

    print(f"Converted dataset to GAR: {graph_yml}")


if __name__ == "__main__":
    main()
