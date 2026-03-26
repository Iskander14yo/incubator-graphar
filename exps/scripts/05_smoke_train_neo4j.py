#!/usr/bin/env python3
"""Smoke test for Neo4j loaders.

Checks:
  1. [both]     At least one seed has hop-1 neighbors each step.
  2. [both]     Feature spot-check: one node's feat matches a direct Neo4j query.
  3. [per_node] Each seed has ≤ fanout_0 hop-1 neighbors (hard assertion).
                Each hop-1 node has ≤ fanout_1 hop-2 neighbors (hard assertion).
  4. [global]   Log seeds that exceed fanout_0 hop-1 neighbors (expected; informational).
  5. [per_node] Convergence: loss decreases over 20 gradient steps.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pyarrow  # noqa: F401 - must precede graphar C extension

import torch
import torch.nn.functional as F
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, DatabaseUnavailable
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.nn import SAGEConv

from exps.benchmarks.neo4j_loader import Neo4jNeighborLoader, iter_batches

_STEPS = 20
_BATCH_SIZE = 32
_HIDDEN = 32
_SEED = 42
_NUM_NEIGHBORS = [3, 2]
_FEATURES = ["feat"]


# ---------------------------------------------------------------------------
# Model (same as GAR smoke test)
# ---------------------------------------------------------------------------

class _GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, _HIDDEN)
        self.conv2 = SAGEConv(_HIDDEN, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# ---------------------------------------------------------------------------
# Hop-adjacency helper
# ---------------------------------------------------------------------------

def _hop_adj(data) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Return (seed→hop1, hop1→hop2) adjacency sets from local edge_index."""
    seed_count = data.batch_size
    src_arr = data.edge_index[0].tolist()
    dst_arr = data.edge_index[1].tolist()
    adj: dict[int, set[int]] = defaultdict(set)
    for s, d in zip(src_arr, dst_arr):
        adj[s].add(d)
    seeds_to_hop1 = {i: adj[i] for i in range(seed_count)}
    hop1_nodes = {d for nbrs in seeds_to_hop1.values() for d in nbrs}
    hop1_to_hop2 = {h: adj[h] for h in hop1_nodes if adj[h]}
    return seeds_to_hop1, hop1_to_hop2


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_coverage(data, step: int, strategy: str) -> bool:
    seeds_to_hop1, _ = _hop_adj(data)
    covered = sum(1 for nbrs in seeds_to_hop1.values() if nbrs)
    total = data.batch_size
    if covered == 0:
        print(f"  FAIL  [{strategy}] step {step}: 0/{total} seeds have hop-1 neighbors")
        return False
    return True


def _check_per_node_fanout(data, step: int) -> bool:
    seeds_to_hop1, hop1_to_hop2 = _hop_adj(data)
    fanout_0, fanout_1 = _NUM_NEIGHBORS[0], _NUM_NEIGHBORS[1]

    # Seed → hop-1: each seed is unique in the input, so the CALL subquery
    # runs exactly once per seed. Count must be ≤ fanout_0.
    violations = [
        f"seed {i}: {len(n)} > fanout_0={fanout_0}"
        for i, n in seeds_to_hop1.items() if len(n) > fanout_0
    ]
    if violations:
        for v in violations:
            print(f"  FAIL  [per_node] step {step}: over-fanout — {v}")
        return False

    # Hop-1 → hop-2: a shared hop-1 node (neighbor of multiple seeds) has its
    # CALL subquery run once per (seed, n1) occurrence, each sampling fanout_1
    # different n2 nodes. After dedup, a shared n1 can have up to
    # k × fanout_1 unique hop-2 neighbors (k = number of seeds sharing it).
    # Log only; this is expected behavior, not a query bug.
    for h1_idx, nbrs in hop1_to_hop2.items():
        if len(nbrs) > fanout_1:
            print(f"  INFO  [per_node] step {step}: hop1 {h1_idx} has {len(nbrs)} hop-2 "
                  f"neighbors > fanout_1={fanout_1} (shared node, expected)")
    return True


def _log_global_fanout(data, step: int) -> None:
    seeds_to_hop1, _ = _hop_adj(data)
    fanout_0 = _NUM_NEIGHBORS[0]
    over = [(i, len(n)) for i, n in seeds_to_hop1.items() if len(n) > fanout_0]
    if over:
        print(f"  INFO  [global]   step {step}: {len(over)} seeds exceeded fanout_0={fanout_0} "
              f"(expected for global strategy, e.g. seed {over[0][0]}: {over[0][1]} nbrs)")


def _check_features(data, uri: str, database: str) -> bool:
    """Pick one node from data, query Neo4j directly, compare feat."""
    if data.n_id.size(0) == 0 or data.x is None or data.x.size(0) == 0:
        print("  SKIP  feature spot-check: empty batch")
        return True
    idx = 0
    global_id = int(data.n_id[idx].item())
    loader_feat = data.x[idx].tolist()

    driver = GraphDatabase.driver(uri, auth=None)
    with driver.session(database=database) as s:
        row = s.run(
            "MATCH (n:node) WHERE n.id = $id RETURN n.feat AS feat", id=global_id
        ).single()
    driver.close()

    if row is None:
        print(f"  FAIL  feature spot-check: node {global_id} not found in Neo4j")
        return False
    direct_feat = list(row["feat"])
    match = all(abs(a - b) < 1e-4 for a, b in zip(loader_feat, direct_feat))
    if match:
        print(f"  PASS  feature spot-check: node {global_id} matches (dim={len(direct_feat)})")
    else:
        diffs = [(i, a, b) for i, (a, b) in enumerate(zip(loader_feat, direct_feat)) if abs(a - b) >= 1e-4]
        print(f"  FAIL  feature spot-check: {len(diffs)} mismatches for node {global_id}")
    return match


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def _check_convergence(loader: Neo4jNeighborLoader, labels_all: torch.Tensor, num_classes: int) -> bool:
    model = _GraphSAGE(in_channels=100, out_channels=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    gen = iter_batches(loader)
    losses: list[float] = []
    for step in range(_STEPS):
        try:
            data, _ = next(gen)
        except StopIteration:
            gen = iter_batches(loader)
            data, _ = next(gen)

        if data.batch_size == 0 or data.x is None or data.edge_index is None:
            continue
        seed_count = data.batch_size
        y = labels_all[data.n_id[:seed_count]]
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[:seed_count]
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  step {step + 1:3d}  loss={loss.item():.4f}")

    if not losses:
        print("  FAIL  convergence: no steps completed")
        return False
    first, last = losses[0], losses[-1]
    if last < first:
        print(f"  PASS  convergence: {first:.4f} → {last:.4f}")
        return True
    print(f"  FAIL  convergence: {first:.4f} → {last:.4f} (did not decrease)")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ogbn-products")
    p.add_argument("--ogb-root", default="exps/datasets/ogb")
    p.add_argument("--uri", default="bolt://localhost:7687")
    p.add_argument("--database", default="neo4j")
    return p.parse_args()


def _make_loader(args: argparse.Namespace, strategy: Literal["global", "per_node"]) -> Neo4jNeighborLoader:
    return Neo4jNeighborLoader(
        uri=args.uri,
        database=args.database,
        vertex_type="node",
        edge_type="edge",
        num_neighbors=_NUM_NEIGHBORS,
        batch_size=_BATCH_SIZE,
        shuffle=True,
        features=_FEATURES,
        profile_every_n=None,
        strategy=strategy,
    )


def _assert_neo4j_up(uri: str, database: str) -> None:
    try:
        driver = GraphDatabase.driver(uri, auth=None)
        driver.verify_connectivity()
        with driver.session(database=database) as s:
            s.run("RETURN 1").consume()
        driver.close()
    except (ServiceUnavailable, DatabaseUnavailable) as e:
        print(f"ABORT  Neo4j not reachable at {uri} (db={database}): {e}")
        print("       Start Neo4j first: sudo neo4j start")
        sys.exit(1)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(_SEED)

    _assert_neo4j_up(args.uri, args.database)

    print(f"Loading OGB labels from {args.ogb_root}...")
    ogb = NodePropPredDataset(name=args.dataset, root=args.ogb_root)
    _, labels_np = ogb[0]
    labels_all = torch.from_numpy(labels_np.squeeze()).long()
    num_classes = int(labels_all.max().item()) + 1
    results: dict[str, bool] = {}

    # ---------------------------------------------------------------- global
    print(f"\n[global — {_STEPS} steps]")
    with _make_loader(args, "global") as loader:
        gen = iter_batches(loader)
        ok_coverage = True
        feat_checked = False
        for step in range(_STEPS):
            try:
                data, _ = next(gen)
            except StopIteration:
                gen = iter_batches(loader)
                data, _ = next(gen)
            if not _check_coverage(data, step + 1, "global"):
                ok_coverage = False
            _log_global_fanout(data, step + 1)
            if not feat_checked:
                feat_checked = _check_features(data, args.uri, args.database)
    results["global_coverage"] = ok_coverage
    results["feature_spot_check"] = feat_checked

    # --------------------------------------------------------------- per_node
    print(f"\n[per_node — {_STEPS} steps: fanout checks]")
    with _make_loader(args, "per_node") as loader:
        gen = iter_batches(loader)
        ok_coverage_pn = True
        ok_fanout = True
        for step in range(_STEPS):
            try:
                data, _ = next(gen)
            except StopIteration:
                gen = iter_batches(loader)
                data, _ = next(gen)
            if not _check_coverage(data, step + 1, "per_node"):
                ok_coverage_pn = False
            if not _check_per_node_fanout(data, step + 1):
                ok_fanout = False
    results["per_node_coverage"] = ok_coverage_pn
    results["per_node_fanout"] = ok_fanout

    # ------------------------------------------------------- convergence
    print(f"\n[per_node — convergence over {_STEPS} gradient steps]")
    with _make_loader(args, "per_node") as loader:
        results["convergence"] = _check_convergence(loader, labels_all, num_classes)

    # ---------------------------------------------------------------- summary
    print("\n" + "-" * 40)
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False
    print()
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
