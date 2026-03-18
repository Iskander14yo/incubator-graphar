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
from graphar_pyspark.enums import FileType
from graphar_pyspark.graph import GraphWriter


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
    parser.add_argument("--spark-driver-memory", default="16g")
    parser.add_argument(
        "--vertex-write-batch-size",
        type=int,
        default=250_000,
        help="Rows per parquet write batch for vertex table.",
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


def _write_vertices_parquet_chunked(
    path: Path,
    node_feat: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> None:
    num_nodes, feat_dim = node_feat.shape
    fields = [pa.field("id", pa.int64())]
    for idx in range(feat_dim):
        fields.append(pa.field(f"feat_{idx:03d}", pa.float64()))
    fields.append(pa.field("label", pa.int64()))
    schema = pa.schema(fields)
    if path.exists():
        path.unlink()

    writer = pq.ParquetWriter(path, schema=schema)
    try:
        labels_1d = labels.reshape(-1)
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            ids = pa.array(np.arange(start, end, dtype=np.int64), type=pa.int64())
            feat_chunk = np.asarray(node_feat[start:end], dtype=np.float64)
            label = pa.array(labels_1d[start:end].astype(np.int64, copy=False), type=pa.int64())

            arrays: list[pa.Array] = [ids]
            for idx in range(feat_dim):
                arrays.append(pa.array(feat_chunk[:, idx], type=pa.float64()))
            arrays.append(label)

            table = pa.Table.from_arrays(arrays, schema=schema)
            writer.write_table(table)
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
    if graph_yml.exists():
        print(f"GAR output already exists, skipping: {graph_yml}")
        return

    graph, labels = _load_ogb_graph(args.dataset, args.ogb_root)
    node_feat = graph["node_feat"]
    edge_index = graph["edge_index"]

    if node_feat.ndim != 2:
        msg = f"Expected node_feat to be 2D, got shape={node_feat.shape}"
        raise ValueError(msg)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        msg = f"Expected edge_index shape (2, E), got shape={edge_index.shape}"
        raise ValueError(msg)

    tmp_dir = Path(args.tmp_root) / args.dataset
    tmp_dir.mkdir(parents=True, exist_ok=True)
    vertices_parquet = tmp_dir / "vertices.parquet"
    edges_parquet = tmp_dir / "edges.parquet"

    print("Preparing parquet sources for GraphAr import...")
    _write_vertices_parquet_chunked(
        path=vertices_parquet,
        node_feat=node_feat,
        labels=labels,
        batch_size=args.vertex_write_batch_size,
    )
    _write_edges_parquet_chunked(
        path=edges_parquet,
        edge_index=edge_index,
        batch_size=args.edge_write_batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.master(args.spark_master)
        .appName(f"ogb-to-gar-{args.dataset}")
        .config("spark.jars", str(spark_jar_path))
        .config("spark.driver.memory", args.spark_driver_memory)
        .getOrCreate()
    )
    initialize(spark)

    print("Writing GAR files via GraphWriter...")
    vertices_df = spark.read.parquet(str(vertices_parquet))
    edges_df = spark.read.parquet(str(edges_parquet))
    graph_writer = GraphWriter.from_python()
    graph_writer.put_vertex_data("node", vertices_df, "id")
    graph_writer.put_edge_data(("node", "edge", "node"), edges_df)
    graph_writer.write(
        str(output_dir),
        args.dataset,
        args.vertex_chunk_size,
        args.edge_chunk_size,
        FileType.PARQUET,
        "gar/v1",
    )
    spark.stop()

    output_dir.mkdir(parents=True, exist_ok=True)
    if not graph_yml.exists():
        msg = f"GAR conversion completed but graph file not found: {graph_yml}"
        raise FileNotFoundError(msg)

    print(f"Converted dataset to GAR: {graph_yml}")


if __name__ == "__main__":
    main()
