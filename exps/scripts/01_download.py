#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from ogb.nodeproppred import NodePropPredDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OGB node-property dataset.")
    parser.add_argument("--dataset", default="ogbn-products", help="OGB dataset name.")
    parser.add_argument("--root", default="exps/datasets/ogb", help="Dataset cache root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    dataset = NodePropPredDataset(name=args.dataset, root=str(root))
    graph, labels = dataset[0]
    num_nodes = int(graph["num_nodes"])
    num_edges = int(graph["edge_index"].shape[1])
    feat_dim = int(graph["node_feat"].shape[1])
    label_shape = tuple(labels.shape)

    print(f"Downloaded: {args.dataset}")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Feature dim: {feat_dim}")
    print(f"Labels shape: {label_shape}")


if __name__ == "__main__":
    main()
