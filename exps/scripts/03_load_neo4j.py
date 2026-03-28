#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ogb.nodeproppred import NodePropPredDataset

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from benchmark_yaml import DEFAULT_PATH, BenchmarkConfig, load_config  # noqa: E402


def _write_nodes_csv(
    path: Path,
    node_feat: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> None:
    """Write nodes CSV for neo4j-admin import.

    Columns: :ID (import reference), id:long (queryable property),
    feat:float[] (semicolon-separated), label:long, :LABEL.
    :ID and id:long hold the same OGB integer index so that
    WHERE n.id IN $seeds works with integer params after indexing.
    """
    num_nodes = node_feat.shape[0]
    labels_1d = labels.reshape(-1)

    with path.open("w", newline="", buffering=8 << 20) as f:
        f.write(":ID,id:long,feat:float[],label:long,:LABEL\n")
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            batch_feat = node_feat[start:end]
            batch_labels = labels_1d[start:end]

            # np.savetxt formats floats in C — much faster than Python f-strings
            buf = io.BytesIO()
            np.savetxt(buf, batch_feat, fmt="%.6g", delimiter=";")
            feat_lines = buf.getvalue().splitlines()

            lines: list[str] = []
            for i, (feat_bytes, label) in enumerate(zip(feat_lines, batch_labels)):
                node_id = start + i
                lines.append(f"{node_id},{node_id},{feat_bytes.decode()},{int(label)},node")
            f.write("\n".join(lines))
            f.write("\n")

    print(f"Written {num_nodes} nodes to {path}")


def _write_relationships_csv(path: Path, edge_index: np.ndarray, batch_size: int) -> None:
    total_edges = edge_index.shape[1]

    with path.open("wb", buffering=8 << 20) as f:
        f.write(b":START_ID,:END_ID,:TYPE\n")
        for start in range(0, total_edges, batch_size):
            end = min(start + batch_size, total_edges)
            chunk = edge_index[:, start:end].T  # (batch, 2)
            buf = io.BytesIO()
            np.savetxt(buf, chunk, fmt="%d", delimiter=",")
            # Each line is "src,dst\n" → append ",edge" before each newline
            f.write(buf.getvalue().replace(b"\n", b",edge\n"))

    print(f"Written {total_edges} relationships to {path}")


def _import_sentinel(csv_dir: Path) -> Path:
    return csv_dir / ".imported"


def _import_done(csv_dir: Path) -> bool:
    """True if a previous import completed successfully."""
    return _import_sentinel(csv_dir).exists()


def _wait_for_neo4j() -> None:
    print("Waiting for Neo4j to be ready...")
    while True:
        result = subprocess.run(["neo4j", "status"], capture_output=True, text=True)
        if "Neo4j is running" in result.stdout:
            break
        time.sleep(1)
    print("Neo4j is running.")


def _grant_traversal(csv_dir: Path) -> None:
    """Grant neo4j user traversal on each ancestor dir and read on CSV files.

    neo4j-admin runs as the neo4j OS user and cannot access files under a home
    directory it has no execute permission on.
    """
    resolved = csv_dir.resolve()
    # Walk from root down to csv_dir, adding o+x on each directory.
    # check=False — system dirs like / may already be fine; failures are harmless.
    current = Path("/")
    for part in resolved.parts[1:]:
        current = current / part
        subprocess.run(["sudo", "chmod", "o+x", str(current)], check=False)
    subprocess.run(
        ["sudo", "chmod", "o+r", str(resolved / "nodes.csv"), str(resolved / "relationships.csv")],
        check=True,
    )


def _run_import(db_name: str, nodes_csv: Path, rels_csv: Path) -> None:
    print("Stopping Neo4j for bulk import...")
    subprocess.run(["sudo", "neo4j", "stop"], check=True)

    _grant_traversal(nodes_csv.parent)

    # --overwrite-destination fails if schema/index files exist (e.g. after
    # 03b_neo4j_index.sh was run on a previous import). Wipe that subtree
    # first so the importer gets a clean slate; the rest is handled by
    # --overwrite-destination (needed because neo4j is always in the catalog).
    schema_dir = Path(f"/var/lib/neo4j/data/databases/{db_name}/schema")
    if schema_dir.exists():
        subprocess.run(["sudo", "-u", "neo4j", "rm", "-rf", str(schema_dir)], check=True)

    print(f"Running neo4j-admin import into database '{db_name}'...")
    # cwd=/tmp so neo4j user can write import.report there
    subprocess.run(
        [
            "sudo", "-u", "neo4j",
            "neo4j-admin",
            "database",
            "import",
            "full",
            db_name,
            f"--nodes={nodes_csv.resolve()}",
            f"--relationships={rels_csv.resolve()}",
            "--overwrite-destination",
        ],
        check=True,
        cwd="/tmp",
    )

    print("Starting Neo4j...")
    subprocess.run(["sudo", "neo4j", "start"], check=True)
    _wait_for_neo4j()


def import_ogb_to_neo4j(config: BenchmarkConfig) -> None:
    n = config.neo4j
    dataset = config.dataset
    ogb_root = config.ogb_root
    db_name = n.database
    csv_dir = Path(n.csv_root) / dataset
    nodes_csv = csv_dir / "nodes.csv"
    rels_csv = csv_dir / "relationships.csv"

    if n.force:
        for f in (nodes_csv, rels_csv, _import_sentinel(csv_dir)):
            f.unlink(missing_ok=True)
        print("force: cleared CSVs and import sentinel.")

    if nodes_csv.exists() and rels_csv.exists():
        print(f"CSVs already exist at {csv_dir}, skipping CSV generation.")
    else:
        print(f"Loading OGB dataset '{dataset}'...")
        ogb = NodePropPredDataset(name=dataset, root=ogb_root)
        graph, labels_raw = ogb[0]
        node_feat: np.ndarray = graph["node_feat"]
        edge_index: np.ndarray = graph["edge_index"]
        labels = np.asarray(labels_raw)

        csv_dir.mkdir(parents=True, exist_ok=True)
        print("Writing nodes CSV (this may take several minutes)...")
        _write_nodes_csv(nodes_csv, node_feat, labels, n.node_batch_size)
        print("Writing relationships CSV...")
        _write_relationships_csv(rels_csv, edge_index, n.edge_batch_size)

    if n.skip_import:
        print("Skipping neo4j-admin import (neo4j.skip_import).")
        return

    if _import_done(csv_dir):
        print(f"Import sentinel found — database '{db_name}' already imported. Skipping.")
        return

    _run_import(db_name, nodes_csv, rels_csv)
    _import_sentinel(csv_dir).touch()
    print(f"Imported '{dataset}' into Neo4j database '{db_name}'.")


def main() -> None:
    p = argparse.ArgumentParser(description="Load OGB dataset into Neo4j via CSV bulk import.")
    p.add_argument("--config", type=Path, default=DEFAULT_PATH, help="Benchmark YAML.")
    import_ogb_to_neo4j(load_config(p.parse_args().config))


if __name__ == "__main__":
    main()
