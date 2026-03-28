"""Benchmark YAML → typed config (exps/config/benchmark.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_EXPS = Path(__file__).resolve().parent.parent
DEFAULT_PATH = _EXPS / "config" / "benchmark.yaml"


@dataclass(frozen=True)
class GarSection:
    graph_path: str
    vertex_type: str
    edge_type: str
    num_features: int


@dataclass(frozen=True)
class Neo4jSection:
    uri: str
    database: str
    profile_every_n: int | None


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str
    loaders: list[str]
    batch_size: int
    num_neighbors: list[int]
    features: list[str]
    num_runs: int
    shuffle: bool
    seed: int
    ogb_root: str
    gar: GarSection
    neo4j: Neo4jSection

    @property
    def gar_root(self) -> str:
        return str(Path(self.gar.graph_path).parent.parent)


def load_config(path: Path | str | None = None) -> BenchmarkConfig:
    p = Path(path) if path else DEFAULT_PATH
    raw = yaml.safe_load(p.read_text())
    return BenchmarkConfig(
        dataset=raw["dataset"],
        loaders=list(raw["loaders"]),
        batch_size=raw["batch_size"],
        num_neighbors=list(raw["num_neighbors"]),
        features=list(raw["features"]),
        num_runs=raw["num_runs"],
        shuffle=raw["shuffle"],
        seed=raw["seed"],
        ogb_root=raw["ogb_root"],
        gar=GarSection(**raw["gar"]),
        neo4j=Neo4jSection(**raw["neo4j"]),
    )
