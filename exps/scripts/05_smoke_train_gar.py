#!/usr/bin/env python3
"""Convergence smoke test for GARNeighborLoader.

Runs a fixed number of gradient steps (not a full epoch) and checks that
the final loss is lower than the initial loss — enough to confirm the loader
feeds correct features, edges, and labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow  # noqa: F401 - must precede graphar C extension

import torch
import torch.nn.functional as F
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.nn import SAGEConv

import graphar as gar
from graphar.ml.torch import GARNeighborLoader

_FEAT_COLS = [f"f{i:03d}" for i in range(100)]
_STEPS = 20          # gradient steps; dataset-size-independent
_BATCH_SIZE = 32
_HIDDEN = 32
_SEED = 42


class _GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, _HIDDEN)
        self.conv2 = SAGEConv(_HIDDEN, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ogbn-products")
    p.add_argument("--gar-root", default="exps/datasets/gar")
    p.add_argument("--ogb-root", default="exps/datasets/ogb")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(_SEED)

    print(f"Loading OGB labels from {args.ogb_root}...")
    ogb = NodePropPredDataset(name=args.dataset, root=args.ogb_root)
    _, labels_np = ogb[0]
    labels_all = torch.from_numpy(labels_np.squeeze()).long()
    num_classes = int(labels_all.max().item()) + 1
    train_nodes = ogb.get_idx_split()["train"].tolist()

    graph_yml = Path(args.gar_root) / args.dataset / f"{args.dataset}.graph.yml"
    print(f"Loading GAR graph from {graph_yml}...")
    graph_info = gar.GraphInfo.load(str(graph_yml.resolve()))

    loader = GARNeighborLoader(
        graph_info,
        vertex_type="node",
        edge_type="edge",
        num_neighbors=[3, 2],
        input_nodes=train_nodes,
        batch_size=_BATCH_SIZE,
        shuffle=True,
        features=_FEAT_COLS,
    )

    model = _GraphSAGE(in_channels=100, out_channels=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"Running {_STEPS} gradient steps...")
    losses: list[float] = []
    gen = iter(loader)
    model.train()
    for step in range(_STEPS):
        try:
            batch = next(gen)
        except StopIteration:
            gen = iter(loader)
            batch = next(gen)

        seed_count = batch.batch_size
        y = labels_all[batch.n_id[:seed_count]]
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:seed_count]
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  step {step + 1:3d}  loss={loss.item():.4f}")

    first, last = losses[0], losses[-1]
    if last < first:
        print(f"\nPASS  loss decreased: {first:.4f} → {last:.4f}")
    else:
        print(f"\nFAIL  loss did not decrease: {first:.4f} → {last:.4f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
