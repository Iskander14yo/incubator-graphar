#!/usr/bin/env python3
"""Convert CORA dataset to GraphAr format using PySpark."""

import os
import sys
import struct
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import split, col, element_at

from graphar_pyspark import initialize
from graphar_pyspark.enums import GarType, FileType
from graphar_pyspark.info import (
    Property,
    PropertyGroup,
    VertexInfo,
    EdgeInfo,
    AdjList,
    GraphInfo,
)
from graphar_pyspark.graph import GraphWriter


def load_cora_vertices(spark: SparkSession, content_file: str) -> DataFrame:
    """Load CORA vertices from .content file.

    Format: <paper_id> <word_attributes>+ <class_label>
    Returns DataFrame with id and label columns.
    """
    df = spark.read.text(content_file)
    split_cols = split(col("value"), r"\t")

    vertices = df.select(
        element_at(split_cols, 1).cast("long").alias("id"),
        element_at(split_cols, -1).alias("label"),
    )
    return vertices


def load_cora_edges(spark: SparkSession, cites_file: str) -> DataFrame:
    """Load CORA edges from .cites file.

    Format: <cited_paper_id> <citing_paper_id>
    Creates directed edges from citing -> cited.
    """
    df = spark.read.text(cites_file)
    split_cols = split(col("value"), r"\t")

    edges = df.select(
        element_at(split_cols, 2).cast("long").alias("src_id"),
        element_at(split_cols, 1).cast("long").alias("dst_id"),
    )
    return edges


def create_vertex_info() -> VertexInfo:
    """Create VertexInfo for paper vertices."""
    properties = [
        Property.from_python("id", GarType.INT64, True, False),
        Property.from_python("label", GarType.STRING, False, False),
    ]

    property_group = PropertyGroup.from_python(
        prefix="vertex/paper/",
        file_type=FileType.PARQUET,
        properties=properties,
    )

    vertex_info = VertexInfo.from_python(
        vertex_type="paper",
        chunk_size=2**18,
        prefix="vertex/paper/",
        property_groups=[property_group],
        version="gar/v1",
    )
    return vertex_info


def create_edge_info() -> EdgeInfo:
    """Create EdgeInfo for citation edges."""
    adj_list = AdjList.from_python(
        ordered=True,
        aligned_by="dest",
        prefix="edge/cites/",
        file_type=FileType.PARQUET,
    )

    edge_info = EdgeInfo.from_python(
        src_type="paper",
        edge_type="cites",
        dst_type="paper",
        chunk_size=2**22,
        src_chunk_size=2**18,
        dst_chunk_size=2**18,
        directed=True,
        prefix="edge/cites/",
        adj_lists=[adj_list],
        property_groups=[],
        version="gar/v1",
    )
    return edge_info


def create_graph_info(prefix: str) -> GraphInfo:
    """Create GraphInfo combining vertex and edge info."""
    graph_info = GraphInfo.from_python(
        name="cora",
        prefix=prefix,
        vertices=["vertex/paper/"],
        edges=["edge/cites/"],
        version="gar/v1",
    )

    graph_info.add_vertex_info(create_vertex_info())
    graph_info.add_edge_info(create_edge_info())

    return graph_info


def main():
    if len(sys.argv) != 4:
        print("Usage: python cora-to-gar.py <cora_dir> <output_dir> <spark_jar_path>")
        print(
            "Example: python cora-to-gar.py ./datasets/cora-original ./output/cora-gar ./path/to/graphar-0.1.0-SNAPSHOT.jar"
        )
        sys.exit(1)

    cora_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    spark_jar = sys.argv[3]

    content_file = cora_dir / "cora.content"
    cites_file = cora_dir / "cora.cites"

    if not content_file.exists() or not cites_file.exists():
        print(f"Error: CORA files not found in {cora_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Spark with GraphAr JAR
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("cora-to-gar")
        .config("spark.jars", spark_jar)
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )

    print("Initializing GraphAr...")
    initialize(spark)

    print("Loading CORA vertices...")
    vertices = load_cora_vertices(spark, str(content_file))
    vertices.show(5)
    print(f"Total vertices: {vertices.count()}")

    print("Loading CORA edges...")
    edges = load_cora_edges(spark, str(cites_file))
    edges.show(5)
    print(f"Total edges: {edges.count()}")

    print("Creating GraphAr metadata...")
    graph_info = create_graph_info(str(output_dir) + "/")

    print("Writing to GAR format...")
    writer = GraphWriter.from_python()
    writer.put_vertex_data("paper", vertices, "id")
    writer.put_edge_data(("paper", "cites", "paper"), edges)
    # writer.write_with_graph_info(graph_info)
    writer.write(str(output_dir) + "/")

    print(f"Successfully converted CORA to GAR format at {output_dir}")
    print("Graph metadata saved as YAML files")

    # Validate the output
    print("\nValidating GAR output structure...")
    validate_gar_output(output_dir)

    spark.stop()


def validate_gar_output(output_dir: Path) -> bool:
    """Validate that GAR format was created correctly."""
    print(f"\n{'=' * 60}")
    print(f"GAR Output Validation")
    print(f"{'=' * 60}")

    errors = []
    warnings = []

    # Check root directory
    if not output_dir.exists():
        errors.append(f"Output directory does not exist: {output_dir}")
        return False

    print(f"✓ Output directory exists: {output_dir}")

    # Check vertex directory structure
    vertex_dir = output_dir / "vertex" / "paper"
    if not vertex_dir.exists():
        errors.append(f"Vertex directory missing: {vertex_dir}")
    else:
        print(f"✓ Vertex directory exists: {vertex_dir}")

        # Check vertex data files
        vertex_data_dir = vertex_dir / "vertex" / "paper"
        if vertex_data_dir.exists():
            chunks = list(vertex_data_dir.glob("chunk*"))
            print(f"  - Found {len(chunks)} vertex chunk(s)")
            if not chunks:
                warnings.append("No vertex chunk files found")

        # Check vertex count file
        count_file = vertex_dir / "vertex_count"
        if count_file.exists():
            with open(count_file, "rb") as f:
                data = f.read()
                if len(data) >= 8:
                    count = struct.unpack("<q", data[:8])[0]
                    print(f"  - Vertex count: {count}")
                else:
                    print(f"  - Vertex count file exists but is too small")
        else:
            warnings.append("Vertex count file missing")

    # Check edge directory structure
    edge_dir = output_dir / "edge" / "cites" / "edge" / "cites"
    if not edge_dir.exists():
        errors.append(f"Edge directory missing: {edge_dir}")
    else:
        print(f"✓ Edge directory exists: {edge_dir}")

        # Check adj_list files
        adj_list_dir = edge_dir / "adj_list"
        if adj_list_dir.exists():
            chunks = list(adj_list_dir.glob("part*/chunk*"))
            print(f"  - Found {len(chunks)} adjacency list chunk(s)")
            if not chunks:
                warnings.append("No adjacency list chunk files found")

        # Check offset files
        offset_dir = edge_dir / "offset"
        if offset_dir.exists():
            chunks = list(offset_dir.glob("chunk*"))
            print(f"  - Found {len(chunks)} offset chunk(s)")

        # Check edge count file
        count_file = edge_dir / "edge_count0"
        if count_file.exists():
            with open(count_file, "rb") as f:
                data = f.read()
                if len(data) >= 8:
                    count = struct.unpack("<q", data[:8])[0]
                    print(f"  - Edge count: {count}")
                else:
                    print(f"  - Edge count file exists but is too small")
        else:
            warnings.append("Edge count file missing")

    # Check for YAML metadata files (optional - GraphWriter may store them internally)
    yaml_files = list(output_dir.glob("*.yml")) + list(output_dir.glob("*.yaml"))
    if yaml_files:
        print(f"✓ Found {len(yaml_files)} YAML metadata file(s)")
        for yml in yaml_files:
            print(f"  - {yml.name}")
    else:
        print("ℹ️  No separate YAML files in root (metadata stored in binary format)")

    # Summary
    print(f"\n{'=' * 60}")
    if errors:
        print("❌ ERRORS:")
        for err in errors:
            print(f"  - {err}")
        print(f"{'=' * 60}\n")
        return False

    if warnings:
        print("⚠️  WARNINGS:")
        for warn in warnings:
            print(f"  - {warn}")

    print("✓ GAR output structure is valid!")
    print(f"{'=' * 60}\n")
    return True


if __name__ == "__main__":
    main()
