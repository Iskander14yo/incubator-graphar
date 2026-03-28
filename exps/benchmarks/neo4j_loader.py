from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from collections.abc import Iterator
from typing import Literal

import torch
from neo4j import GraphDatabase
from torch_geometric.data import Data

from .timings import BatchTimings, _Timer

# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def _return_clause(num_hops: int, features: list[str]) -> str:
    parts = ["n0.id AS id_0"]
    for f in features:
        parts.append(f"n0.{f} AS {f}_0")
    for i in range(1, num_hops + 1):
        parts.append(f"n{i}.id AS id_{i}")
        for f in features:
            parts.append(f"n{i}.{f} AS {f}_{i}")
    return "RETURN " + ", ".join(parts)


def _global_query(
    vertex_type: str,
    edge_type: str,
    num_hops: int,
    features: list[str],
    profile: bool = False,
) -> str:
    # Reproduces the paper's Figure 4: single chained OPTIONAL MATCH so that
    # seeds with no outgoing edges still appear in results (with nulls for
    # hop-1/2 nodes), followed by a global ORDER BY rand() LIMIT.
    prefix = "PROFILE " if profile else ""
    chain = "-".join(
        f"[:{edge_type}]->(n{i + 1}:{vertex_type})" for i in range(num_hops)
    )
    lines = [
        f"{prefix}MATCH (n0:{vertex_type})",
        "WHERE n0.id IN $seed_nodes",
        f"OPTIONAL MATCH (n0)-{chain}",
        f"WITH {', '.join(f'n{i}' for i in range(num_hops + 1))} ORDER BY rand()",
        "LIMIT $limit",
    ]
    lines.append(_return_clause(num_hops, features))
    return "\n".join(lines)


def _per_node_query(
    vertex_type: str,
    edge_type: str,
    num_neighbors: list[int],
    features: list[str],
    profile: bool = False,
) -> str:
    num_hops = len(num_neighbors)
    prefix = "PROFILE " if profile else ""
    lines = [
        f"{prefix}MATCH (n0:{vertex_type})",
        "WHERE n0.id IN $seed_nodes",
    ]
    for i in range(num_hops):
        lines += [
            f"CALL (n{i}) {{",
            f"    MATCH (n{i})-[:{edge_type}]->(n{i + 1}:{vertex_type})",
            f"    WITH n{i + 1} ORDER BY rand() LIMIT $fanout_{i}",
            f"    RETURN collect(n{i + 1}) AS hop{i + 1}",
            "}",
            f"UNWIND hop{i + 1} AS n{i + 1}",
        ]
    lines.append(_return_clause(num_hops, features))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------

def _feat_values(row: dict, hop: int, features: list[str]) -> list[float]:
    parts: list[float] = []
    for f in features:
        val = row[f"{f}_{hop}"]
        if isinstance(val, (list, tuple)):
            parts.extend(float(v) for v in val)
        else:
            parts.append(float(val))
    return parts


def _rows_to_data(
    rows: list[dict],
    num_hops: int,
    features: list[str],
    seed_nodes: list[int],
) -> Data:
    node_feats: dict[int, list[float]] = {}
    edge_set: set[tuple[int, int]] = set()

    for row in rows:
        for hop in range(num_hops + 1):
            raw_id = row[f"id_{hop}"]
            if raw_id is None:
                break  # OPTIONAL MATCH — this hop and all later ones are null
            nid = int(raw_id)
            if nid not in node_feats:
                node_feats[nid] = _feat_values(row, hop, features)
        for hop in range(num_hops):
            if row[f"id_{hop}"] is None or row[f"id_{hop + 1}"] is None:
                break
            src = int(row[f"id_{hop}"])
            dst = int(row[f"id_{hop + 1}"])
            edge_set.add((src, dst))

    seed_set = set(seed_nodes)
    seen_seeds = [nid for nid in seed_nodes if nid in node_feats]
    other_nodes = [nid for nid in node_feats if nid not in seed_set]
    ordered_ids = seen_seeds + other_nodes

    if not ordered_ids:
        x = torch.empty((0, 0), dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        n_id = torch.empty((0,), dtype=torch.long)
        batch = Data(x=x, edge_index=edge_index)
        batch.n_id = n_id
        batch.batch_size = 0
        batch.input_id = torch.tensor(seed_nodes, dtype=torch.long)
        return batch

    id_to_idx = {nid: i for i, nid in enumerate(ordered_ids)}

    x = torch.tensor([node_feats[nid] for nid in ordered_ids], dtype=torch.float32)

    valid_edges = [(s, d) for s, d in edge_set if s in id_to_idx and d in id_to_idx]
    if valid_edges:
        edge_index = torch.tensor(
            [[id_to_idx[s] for s, _ in valid_edges], [id_to_idx[d] for _, d in valid_edges]],
            dtype=torch.long,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.n_id = torch.tensor(ordered_ids, dtype=torch.long)
    data.batch_size = len(seen_seeds)
    data.input_id = torch.tensor(seen_seeds, dtype=torch.long)
    return data


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------

def _extract_profile(plan) -> dict:
    """Recursively aggregate stats from a ProfiledPlan tree."""
    db_hits = getattr(plan, "db_hits", 0) or 0
    cache_hits = getattr(plan, "page_cache_hits", 0) or 0
    cache_misses = getattr(plan, "page_cache_misses", 0) or 0
    op_time_us = getattr(plan, "time", 0) or 0

    operators = [{
        "operator": plan.operator_type,
        "db_hits": db_hits,
        "page_cache_hits": cache_hits,
        "page_cache_misses": cache_misses,
        "time_us": op_time_us,
    }]

    for child in (plan.children or []):
        child_data = _extract_profile(child)
        db_hits += child_data["total_db_hits"]
        cache_hits += child_data["total_page_cache_hits"]
        cache_misses += child_data["total_page_cache_misses"]
        operators.extend(child_data["operators"])

    return {
        "total_db_hits": db_hits,
        "total_page_cache_hits": cache_hits,
        "total_page_cache_misses": cache_misses,
        "operators": operators,
    }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class Neo4jNeighborLoader:
    """Neighbor loader backed by Neo4j.

    Supports two sampling strategies:
    - ``"global"``: single flat multi-hop MATCH with a global ORDER BY rand() LIMIT.
    - ``"per_node"``: correlated subqueries (Neo4j 5.x+) giving each seed exactly
      ``num_neighbors[k]`` sampled neighbors per hop.
    """

    def __init__(
        self,
        uri: str,
        database: str,
        vertex_type: str,
        edge_type: str,
        num_neighbors: list[int],
        batch_size: int = 128,
        shuffle: bool = True,
        features: list[str] | None = None,
        profile_every_n: int | None = 50,
        strategy: Literal["global", "per_node"] = "global",
    ) -> None:
        if not num_neighbors:
            msg = "num_neighbors must not be empty"
            raise ValueError(msg)

        self._driver = GraphDatabase.driver(uri, auth=None)
        self._database = database
        self._num_neighbors = list(num_neighbors)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._features = features or []
        self._profile_every_n = profile_every_n
        self._strategy = strategy

        num_hops = len(num_neighbors)

        if strategy == "per_node":
            self._query = _per_node_query(vertex_type, edge_type, num_neighbors, self._features)
            self._query_profile = _per_node_query(
                vertex_type, edge_type, num_neighbors, self._features, profile=True
            )
        else:
            self._query = _global_query(vertex_type, edge_type, num_hops, self._features)
            self._query_profile = _global_query(
                vertex_type, edge_type, num_hops, self._features, profile=True
            )

        with self._driver.session(database=database) as s:
            q = f"MATCH (n:{vertex_type}) RETURN count(n) AS cnt"  # noqa: S608
            cnt = s.run(q).single()["cnt"]  # type: ignore[index]
            self._input_nodes = list(range(int(cnt)))

        # limit for global strategy: batch_size × ∏(num_neighbors)
        self._global_limit = batch_size * math.prod(num_neighbors)
        self.timings: defaultdict = defaultdict(list)

    def _params(self, seed_nodes: list[int]) -> dict:
        params: dict = {"seed_nodes": seed_nodes}
        if self._strategy == "global":
            params["limit"] = self._global_limit
        else:
            for i, fanout in enumerate(self._num_neighbors):
                params[f"fanout_{i}"] = fanout
        return params

    def _iter_seed_batches(self) -> Iterator[list[int]]:
        nodes = self._input_nodes[:]
        if self._shuffle:
            random.shuffle(nodes)
        for start in range(0, len(nodes), self._batch_size):
            yield nodes[start : start + self._batch_size]

    def _run_batch(
        self, seed_nodes: list[int], use_profile: bool
    ) -> tuple[Data, dict | None]:
        """Returns (data, profile_dict|None)."""
        query = self._query_profile if use_profile else self._query
        params = self._params(seed_nodes)
        num_hops = len(self._num_neighbors)

        with _Timer("retrieval", self.timings):
            with self._driver.session(database=self._database) as session:
                result = session.run(query, params)  # type: ignore[arg-type]
                raw_rows = list(result)  # drain cursor; data transfer happens here
                profile_data = None
                if use_profile:
                    summary = result.consume()
                    if summary.profile:
                        profile_data = _extract_profile(summary.profile)

        with _Timer("conversion", self.timings):
            rows = [dict(r) for r in raw_rows]
            data = _rows_to_data(rows, num_hops, self._features, seed_nodes)

        return data, profile_data

    def __len__(self) -> int:
        if not self._input_nodes:
            return 0
        return (len(self._input_nodes) + self._batch_size - 1) // self._batch_size

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# iter_batches
# ---------------------------------------------------------------------------

def iter_batches(loader: Neo4jNeighborLoader) -> Iterator[tuple[Data, BatchTimings]]:
    loader.timings = defaultdict(list)
    t = loader.timings
    profile_every_n = loader._profile_every_n

    for batch_id, seed_nodes in enumerate(loader._iter_seed_batches()):
        use_profile = profile_every_n is not None and batch_id % profile_every_n == 0

        t_start = time.perf_counter()
        data, profile_data = loader._run_batch(seed_nodes, use_profile=use_profile)
        total_ms = (time.perf_counter() - t_start) * 1000

        yield data, BatchTimings(
            batch_id=batch_id,
            total_ms=total_ms,
            retrieval_ms=t["retrieval"][-1] * 1000,
            conversion_ms=t["conversion"][-1] * 1000,
            sampled_nodes=int(data.n_id.size(0)),
            sampled_edges=int(data.edge_index.size(1)) if data.edge_index is not None else 0,
            neo4j_profile=profile_data,
        )
