#!/usr/bin/env python3
"""Stage 4 benchmark runner."""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import pyarrow  # noqa: F401 - must precede graphar C extension
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))  # exps/ → enables `from benchmarks.xxx`

import graphar as gar
from benchmarks.gar_loader import iter_batches as _gar_iter
from benchmarks.neo4j_loader import Neo4jNeighborLoader, iter_batches as _neo4j_iter
from benchmarks.pyg_loader import PyGNeighborLoader, iter_batches as _pyg_iter
from benchmarks.timings import BatchTimings, SystemSample
from graphar.ml.torch import GARNeighborLoader


# ---------------------------------------------------------------------------
# Hardware fingerprint
# ---------------------------------------------------------------------------

def _hardware_info() -> dict:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    disk_type = "unknown"
    try:
        out = subprocess.run(
            ["lsblk", "-d", "-o", "NAME,ROTA"], capture_output=True, text=True, check=False
        ).stdout
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                disk_type = "ssd/nvme" if parts[1] == "0" else "hdd"
                break
    except Exception:
        pass

    return {
        "cpu_model": cpu_model,
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "disk_type": disk_type,
    }


# ---------------------------------------------------------------------------
# Background system monitor
# ---------------------------------------------------------------------------

class _SystemMonitor:
    """Samples CPU %, RSS, and disk read throughput at ~1 s intervals."""

    def __init__(self) -> None:
        self._proc = psutil.Process()
        self._samples: list[SystemSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[SystemSample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        return list(self._samples)

    def _run(self) -> None:
        prev_disk = psutil.disk_io_counters()
        prev_t = time.perf_counter()
        while not self._stop.wait(1.0):
            now = time.perf_counter()
            cur_disk = psutil.disk_io_counters()
            dt = max(now - prev_t, 1e-9)
            read_bytes = (
                (cur_disk.read_bytes - prev_disk.read_bytes)
                if cur_disk and prev_disk
                else 0
            )
            try:
                cpu = self._proc.cpu_percent()
                rss = self._proc.memory_info().rss / 1e6
            except psutil.NoSuchProcess:
                break
            self._samples.append(SystemSample(
                timestamp_ms=int((now - self._t0) * 1000),
                cpu_pct=cpu,
                rss_mb=round(rss, 1),
                disk_read_mb_s=round(read_bytes / dt / 1e6, 2),
            ))
            prev_disk = cur_disk
            prev_t = now


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _clear_caches() -> None:
    script = Path(__file__).parent / "clear_caches.sh"
    subprocess.run(["bash", str(script)], check=True)


def _ogb_train_split(dataset: str, ogb_root: str) -> list[int]:
    from ogb.nodeproppred import NodePropPredDataset
    ds = NodePropPredDataset(name=dataset, root=ogb_root)
    return ds.get_idx_split()["train"].tolist()


# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------

def _make_gar_loader(cfg: dict) -> GARNeighborLoader:
    gar_cfg = cfg["gar"]
    graph_info = gar.GraphInfo.load(str(Path(gar_cfg["graph_path"]).resolve()))
    num_features = gar_cfg.get("num_features", 100)
    features = [f"f{i:03d}" for i in range(num_features)]
    return GARNeighborLoader(
        graph_info,
        vertex_type=gar_cfg["vertex_type"],
        edge_type=gar_cfg["edge_type"],
        num_neighbors=cfg["num_neighbors"],
        batch_size=cfg["batch_size"],
        shuffle=cfg.get("shuffle", False),
        features=features,
    )


def _make_neo4j_loader(cfg: dict, loader_name: str) -> Neo4jNeighborLoader:
    neo4j_cfg = cfg["neo4j"]
    strategy = "global" if loader_name == "neo4j-global" else "per_node"
    return Neo4jNeighborLoader(
        uri=neo4j_cfg["uri"],
        database=neo4j_cfg["database"],
        vertex_type="node",
        edge_type="edge",
        num_neighbors=cfg["num_neighbors"],
        batch_size=cfg["batch_size"],
        shuffle=cfg.get("shuffle", False),
        features=cfg.get("features", ["feat"]),
        profile_every_n=neo4j_cfg.get("profile_every_n", 50),
        strategy=strategy,
    )


def _make_pyg_loader(cfg: dict) -> PyGNeighborLoader:
    return PyGNeighborLoader(
        dataset_name=cfg["dataset"],
        ogb_root=cfg.get("ogb_root", "exps/datasets/ogb"),
        num_neighbors=cfg["num_neighbors"],
        # input_nodes=torch.tensor(train_nodes, dtype=torch.long),
        batch_size=cfg["batch_size"],
        shuffle=cfg.get("shuffle", False),
    )


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def _run_epoch(
    loader, iter_fn, monitor: _SystemMonitor
) -> tuple[list[BatchTimings], list[SystemSample]]:
    monitor.start()
    batch_timings: list[BatchTimings] = []
    for _batch, bt in iter_fn(loader):
        batch_timings.append(bt)
    system_samples = monitor.stop()
    return batch_timings, system_samples


# ---------------------------------------------------------------------------
# Per-loader orchestration
# ---------------------------------------------------------------------------

def _run_loader(
    loader_name: str,
    loader,
    iter_fn,
    num_runs: int,
    result_dir: Path,
) -> None:
    monitor = _SystemMonitor()
    runs = []

    for run_id in range(num_runs):
        run_type = "cold" if run_id == 0 else "warm"
        print(f"  [{loader_name}] run {run_id} ({run_type})...", flush=True)

        if run_type == "cold":
            try:
                _clear_caches()
            except Exception as e:
                print(f"  WARNING: cache clear failed: {e}", flush=True)

        batch_timings, system_samples = _run_epoch(loader, iter_fn, monitor)
        mean_ms = (
            sum(bt.total_ms for bt in batch_timings) / len(batch_timings)
            if batch_timings else 0.0
        )
        print(f"    {len(batch_timings)} batches, mean={mean_ms:.1f} ms", flush=True)

        runs.append({
            "run_id": run_id,
            "type": run_type,
            "batches": [dataclasses.asdict(bt) for bt in batch_timings],
            "system_metrics": [dataclasses.asdict(ss) for ss in system_samples],
        })

    out = result_dir / f"{loader_name}.json"
    out.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"  → {out}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="exps/config/benchmark.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    loaders_to_run = cfg["loaders"]
    num_runs = cfg["num_runs"]
    dataset = cfg["dataset"]
    ogb_root = cfg["ogb_root"]
    seed = cfg["seed"]

    torch.manual_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    result_dir = Path("exps/results") / dataset / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    sha = _git_sha()
    run_info = {
        "timestamp": timestamp,
        "git_sha": sha,
        "dataset": dataset,
        "batch_size": cfg["batch_size"],
        "num_neighbors": cfg["num_neighbors"],
        "seed": seed,
        "num_runs": num_runs,
        "shuffle": cfg.get("shuffle", False),
        "hardware": _hardware_info(),
        "notes": "",
    }
    (result_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Results dir: {result_dir}")

    for loader_name in loaders_to_run:
        print(f"\n=== {loader_name} ===", flush=True)
        loader = None
        try:
            if loader_name == "gar":
                loader = _make_gar_loader(cfg)
                iter_fn = _gar_iter
            elif loader_name in ("neo4j-global", "neo4j-per-node"):
                loader = _make_neo4j_loader(cfg, loader_name)
                iter_fn = _neo4j_iter
            elif loader_name == "pyg-inmem":
                loader = _make_pyg_loader(cfg)
                iter_fn = _pyg_iter
            else:
                print(f"  Unknown loader '{loader_name}', skipping.")
                continue

            _run_loader(loader_name, loader, iter_fn, num_runs, result_dir)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
        finally:
            if loader is not None and hasattr(loader, "close"):
                loader.close()

    manifest = Path("exps/results/runs_manifest.jsonl")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a") as f:
        f.write(json.dumps({
            "timestamp": timestamp,
            "dataset": dataset,
            "git_sha": sha,
            "result_dir": str(result_dir) + "/",
            "notes": "",
        }) + "\n")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
