#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ogb.nodeproppred import NodePropPredDataset

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from benchmark_yaml import DEFAULT_PATH, BenchmarkConfig, load_config  # noqa: E402


def download(config: BenchmarkConfig) -> None:
    root = Path(config.ogb_root)
    root.mkdir(parents=True, exist_ok=True)
    ds = NodePropPredDataset(name=config.dataset, root=str(root))
    graph, labels = ds[0]
    num_nodes = int(graph["num_nodes"])
    num_edges = int(graph["edge_index"].shape[1])
    feat_dim = int(graph["node_feat"].shape[1])
    label_shape = tuple(labels.shape)

    print(f"Downloaded: {config.dataset}")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Feature dim: {feat_dim}")
    print(f"Labels shape: {label_shape}")


def main() -> None:
    p = argparse.ArgumentParser(description="Download OGB node-property dataset.")
    p.add_argument("--config", type=Path, default=DEFAULT_PATH, help="Benchmark YAML.")
    download(load_config(p.parse_args().config))


if __name__ == "__main__":
    main()
