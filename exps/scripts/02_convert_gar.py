#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ogb.nodeproppred import NodePropPredDataset
from pyspark.sql import SparkSession
from graphar_pyspark import initialize
from graphar_pyspark.enums import AdjListType, FileType, GarType
from graphar_pyspark.info import AdjList, EdgeInfo, GraphInfo, Property, PropertyGroup, VertexInfo
from graphar_pyspark.writer import EdgeWriter, VertexWriter


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
        "--spark-jar",
        default=None,
        help="Path to GraphAr Spark JAR. Can also be set via GRAPHAR_SPARK_JAR.",
    )
    parser.add_argument("--spark-master", default="local[*]")
    parser.add_argument("--spark-driver-memory", default="8g")
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
    """Write all vertex columns to a single flat Parquet for Spark import."""
    num_nodes = node_feat.shape[0]
    feat_dim = node_feat.shape[1]
    labels_1d = labels.reshape(-1)

    col_names = ["_graphArVertexIndex", "id"] + [f"f{i:03d}" for i in range(feat_dim)] + ["label"]
    schema = pa.schema(
        [("_graphArVertexIndex", pa.int64()), ("id", pa.int64())]
        + [(name, pa.float32()) for name in col_names[2:-1]]
        + [("label", pa.int64())]
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
                [pa.array(idx), pa.array(idx)]
                + [pa.array(feat_T[i]) for i in range(feat_dim)]
                + [pa.array(labels_1d[start:end].astype(np.int64, copy=False))]
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
            table = pa.Table.from_arrays([src, dst], schema=schema)
            writer.write_table(table)
    finally:
        writer.close()


def _detect_local_spark_jar() -> Path | None:
    target_dir = Path("maven-projects/spark/graphar/target")
    if not target_dir.exists():
        return None
    skip_suffixes = ("-sources.jar", "-javadoc.jar")
    jars = [
        j for j in target_dir.glob("*.jar")
        if not any(j.name.endswith(s) for s in skip_suffixes) and not j.name.startswith("original-")
    ]
    # prefer shaded JAR — it bundles GarDataSource and all transitive deps
    shaded = [j for j in jars if j.name.endswith("-shaded.jar")]
    if shaded:
        return shaded[0]
    return jars[0] if jars else None


def _make_vertex_property_groups(
    vertex_groups: list[list[str]],
    file_type: FileType,
) -> list[PropertyGroup]:
    """Convert column-name groups to GraphAr PropertyGroup objects (requires Spark initialized)."""
    result: list[PropertyGroup] = []
    for group in vertex_groups:
        props = []
        for name in group:
            if name == "id":
                props.append(Property.from_python(name, GarType.INT64, True, False))
            elif name == "label":
                props.append(Property.from_python(name, GarType.INT64, False, True))
            else:
                props.append(Property.from_python(name, GarType.FLOAT, False, True))
        result.append(PropertyGroup.from_python("", file_type, props))
    return result


def _build_graph_info(
    output_dir: Path,
    dataset: str,
    vertex_groups: list[list[str]],
    vertex_chunk_size: int,
    edge_chunk_size: int,
    file_type: FileType,
    version: str,
) -> tuple[GraphInfo, VertexInfo, EdgeInfo]:
    """Build GraphInfo with split vertex property groups and persist YAML info files."""
    property_groups = _make_vertex_property_groups(vertex_groups, file_type)
    vertex_info = VertexInfo.from_python(
        vertex_type="node",
        chunk_size=vertex_chunk_size,
        prefix="vertex/node/",
        property_groups=property_groups,
        version=version,
    )

    adj_lists = [
        AdjList.from_python(True, "src", "ordered_by_source", file_type),
        AdjList.from_python(True, "dst", "ordered_by_dest", file_type),
    ]
    edge_info = EdgeInfo.from_python(
        src_type="node",
        edge_type="edge",
        dst_type="node",
        chunk_size=edge_chunk_size,
        src_chunk_size=vertex_chunk_size,
        dst_chunk_size=vertex_chunk_size,
        directed=True,
        prefix="edge/node_edge_node",
        adj_lists=adj_lists,
        property_groups=[],
        version=version,
    )

    graph_info = GraphInfo.from_python(
        name=dataset,
        prefix=str(output_dir),
        vertices=["node.vertex.yml"],
        edges=["node_edge_node.edge.yml"],
        version=version,
    )
    graph_info.add_vertex_info(vertex_info)
    graph_info.add_edge_info(edge_info)

    (output_dir / "node.vertex.yml").write_text(vertex_info.dump())
    (output_dir / "node_edge_node.edge.yml").write_text(edge_info.dump())
    (output_dir / f"{dataset}.graph.yml").write_text(graph_info.dump())

    return graph_info, vertex_info, edge_info


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

    spark_jar = args.spark_jar or os.environ.get("GRAPHAR_SPARK_JAR")
    if not spark_jar:
        detected = _detect_local_spark_jar()
        if detected is not None:
            spark_jar = str(detected)
    if not spark_jar:
        msg = (
            "Missing GraphAr Spark JAR. Pass --spark-jar, set GRAPHAR_SPARK_JAR, "
            "or run exps/scripts/00_setup.sh to auto-build it."
        )
        raise ValueError(msg)
    spark_jar_path = Path(spark_jar)
    if not spark_jar_path.exists():
        msg = f"Spark JAR not found: {spark_jar_path}"
        raise FileNotFoundError(msg)

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

    print("Writing vertex parquet for Spark import...")
    _write_vertex_parquet_flat(
        path=vertices_parquet,
        node_feat=node_feat,
        labels=labels,
        batch_size=args.vertex_write_batch_size,
    )

    print("Writing edge parquet for Spark import...")
    _write_edges_parquet_chunked(
        path=edges_parquet,
        edge_index=edge_index,
        batch_size=args.edge_write_batch_size,
    )

    spark = (
        SparkSession.builder.master(args.spark_master)
        .appName(f"ogb-to-gar-{args.dataset}")
        .config("spark.jars", str(spark_jar_path))
        .config("spark.driver.memory", args.spark_driver_memory)
        # Each EdgeWriter does an internal SQL sort. The default 200 shuffle
        # output partitions causes each map task to open 200 shuffle files
        # simultaneously → extreme GC pressure on large edge sets. 4 partitions
        # is sufficient; the EdgeWriter re-partitions into vertex chunks anyway.
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:G1HeapRegionSize=32M")
        .getOrCreate()
    )
    initialize(spark)

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_info, vertex_info, edge_info = _build_graph_info(
        output_dir=output_dir,
        dataset=args.dataset,
        vertex_groups=vertex_groups,
        vertex_chunk_size=args.vertex_chunk_size,
        edge_chunk_size=args.edge_chunk_size,
        file_type=FileType.PARQUET,
        version="gar/v1",
    )

    vertex_num = node_feat.shape[0]

    print("Writing vertex chunks via Spark...")
    vertices_df = spark.read.parquet(str(vertices_parquet))
    vertex_writer = VertexWriter.from_python(
        prefix=str(output_dir),
        vertex_info=vertex_info,
        vertex_df=vertices_df,
        num_vertices=vertex_num,
    )
    vertex_writer.write_vertex_properties()

    print("Building edge adjacency lists via Spark...")
    edges_df = (
        spark.read.parquet(str(edges_parquet))
        .withColumnRenamed("src_id", "_graphArSrcIndex")
        .withColumnRenamed("dst_id", "_graphArDstIndex")
    )
    for adj_list_type in (AdjListType.ORDERED_BY_SOURCE, AdjListType.ORDERED_BY_DEST):
        edge_writer = EdgeWriter.from_python(
            prefix=str(output_dir),
            edge_info=edge_info,
            adj_list_type=adj_list_type,
            vertex_num=vertex_num,
            edge_df=edges_df,
        )
        edge_writer.write_edges()
    spark.stop()

    output_dir.mkdir(parents=True, exist_ok=True)
    if not graph_yml.exists():
        msg = f"GAR conversion completed but graph file not found: {graph_yml}"
        raise FileNotFoundError(msg)

    print(f"Converted dataset to GAR: {graph_yml}")


if __name__ == "__main__":
    main()
