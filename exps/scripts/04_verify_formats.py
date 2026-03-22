#!/usr/bin/env python3
"""Smoke-test that GAR and Neo4j stores are correctly loaded and queryable."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# pyarrow must be imported before any graphar C extension to avoid absl symbol conflicts
import pyarrow  # noqa: F401

import graphar as gar
from graphar.importer.data_import import check as gar_check
from graphar.ml.torch import GARNeighborLoader
from neo4j import GraphDatabase


SEED_NODES = [0, 1_000, 100_000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify GAR and Neo4j data loads.")
    parser.add_argument("--dataset", default="ogbn-products")
    parser.add_argument("--gar-root", default="exps/datasets/gar")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-database", default="neo4j")
    return parser.parse_args()


def _pass(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


# ---------------------------------------------------------------------------
# GAR checks
# ---------------------------------------------------------------------------

def verify_gar(dataset: str, gar_root: str) -> bool:
    graph_yml = Path(gar_root) / dataset / f"{dataset}.graph.yml"
    print(f"\n[GAR] {graph_yml}")
    ok = True

    # 1. Metadata integrity
    try:
        result = gar_check(str(graph_yml))
        _pass(f"metadata check: {result}")
    except Exception as e:
        _fail(f"metadata check: {e}")
        return False

    # 2. Load graph info
    try:
        graph_info = gar.GraphInfo.load(str(graph_yml.resolve()))
        _pass("GraphInfo.load()")
    except Exception as e:
        _fail(f"GraphInfo.load(): {e}")
        return False

    # 3. One mini-batch through GARNeighborLoader
    try:
        loader = GARNeighborLoader(
            graph_info,
            vertex_type="node",
            edge_type="edge",
            num_neighbors=[5],
            input_nodes=SEED_NODES,
            batch_size=len(SEED_NODES),
            shuffle=False,
            features=["f000"],
        )
        batch = next(iter(loader))
        assert batch.x is not None and batch.x.shape[0] > 0, "empty x"
        assert batch.edge_index.shape[0] == 2, "bad edge_index shape"
        _pass(
            f"neighbor sample: {batch.x.shape[0]} nodes, "
            f"{batch.edge_index.shape[1]} edges"
        )
    except Exception as e:
        _fail(f"neighbor sample: {e}")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Neo4j checks
# ---------------------------------------------------------------------------

def verify_neo4j(uri: str, database: str) -> bool:
    print(f"\n[Neo4j] {uri}  db={database}")
    ok = True

    try:
        driver = GraphDatabase.driver(uri, auth=None)
        driver.verify_connectivity()
        _pass("connectivity")
    except Exception as e:
        _fail(f"connectivity: {e}")
        return False

    with driver.session(database=database) as session:
        # 1. Node count
        try:
            cnt = session.run("MATCH (n:node) RETURN count(n) AS cnt").single()["cnt"]
            if cnt > 0:
                _pass(f"node count: {cnt:,}")
            else:
                _fail("node count is 0")
                ok = False
        except Exception as e:
            _fail(f"node count: {e}")
            ok = False

        # 2. Index state
        try:
            row = session.run(
                "SHOW INDEXES YIELD name, state WHERE name = 'node_id'"
            ).single()
            if row and row["state"] == "ONLINE":
                _pass(f"index 'node_id' is ONLINE")
            elif row:
                _fail(f"index 'node_id' state: {row['state']}")
                ok = False
            else:
                _fail("index 'node_id' not found")
                ok = False
        except Exception as e:
            _fail(f"index check: {e}")
            ok = False

        # 3. Neighbor query for a few seed nodes
        try:
            result = session.run(
                """
                MATCH (n:node)-[:edge]->(m:node)
                WHERE n.id IN $seeds
                RETURN n.id AS src, count(m) AS neighbor_count
                """,
                seeds=SEED_NODES,
            ).data()
            if result:
                summary = ", ".join(
                    f"node {r['src']} → {r['neighbor_count']} neighbors"
                    for r in result
                )
                _pass(f"neighbor query: {summary}")
            else:
                _fail("neighbor query returned no rows")
                ok = False
        except Exception as e:
            _fail(f"neighbor query: {e}")
            ok = False

    driver.close()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    gar_ok = verify_gar(args.dataset, args.gar_root)
    neo4j_ok = verify_neo4j(args.neo4j_uri, args.neo4j_database)

    print()
    if gar_ok and neo4j_ok:
        print("All checks passed.")
    else:
        failed = [s for s, ok in [("GAR", gar_ok), ("Neo4j", neo4j_ok)] if not ok]
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
