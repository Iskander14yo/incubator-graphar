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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))  # exps/ → enables `from benchmarks.xxx`
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_yaml import DEFAULT_PATH, BenchmarkConfig, load_config  # noqa: E402

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


def _drop_os_cache() -> None:
    """Drop OS page cache only — safe to call for any loader."""
    subprocess.run(
        ["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"], check=True
    )


def _restart_neo4j() -> None:
    """Flush Neo4j page cache by restarting the service."""
    subprocess.run(["sudo", "neo4j", "stop"], check=True)
    subprocess.run(["sudo", "neo4j", "start"], check=True)
    # wait until bolt is ready
    for _ in range(60):
        result = subprocess.run(
            ["neo4j", "status"], capture_output=True, text=True, check=False
        )
        if "Neo4j is running" in result.stdout:
            break
        time.sleep(1)


def _clear_caches(loader_name: str) -> None:
    """Drop OS page cache; also restart Neo4j when running a Neo4j loader."""
    _drop_os_cache()
    if loader_name.startswith("neo4j"):
        _restart_neo4j()


# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------

def _make_gar_loader(config: BenchmarkConfig) -> GARNeighborLoader:
    g = config.gar
    graph_info = gar.GraphInfo.load(str(Path(g.graph_path).resolve()))
    features = [f"f{i:03d}" for i in range(g.num_features)]
    return GARNeighborLoader(
        graph_info,
        vertex_type=g.vertex_type,
        edge_type=g.edge_type,
        num_neighbors=config.num_neighbors,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        features=features,
    )


def _make_neo4j_loader(config: BenchmarkConfig, loader_name: str) -> Neo4jNeighborLoader:
    n = config.neo4j
    strategy = "global" if loader_name == "neo4j-global" else "per_node"
    return Neo4jNeighborLoader(
        uri=n.uri,
        database=n.database,
        vertex_type="node",
        edge_type="edge",
        num_neighbors=config.num_neighbors,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        features=config.features,
        profile_every_n=n.profile_every_n,
        strategy=strategy,
    )


def _make_pyg_loader(config: BenchmarkConfig) -> PyGNeighborLoader:
    return PyGNeighborLoader(
        dataset_name=config.dataset,
        ogb_root=config.ogb_root,
        num_neighbors=config.num_neighbors,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
    )


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def _run_epoch(
    loader, iter_fn, monitor: _SystemMonitor, desc: str
) -> tuple[list[BatchTimings], list[SystemSample]]:
    total = len(loader) # if hasattr(loader, "__len__") else None
    monitor.start()
    batch_timings: list[BatchTimings] = []
    with tqdm(iter_fn(loader), total=total, desc=desc, unit="batch", leave=False) as pbar:
        for _batch, bt in pbar:
            batch_timings.append(bt)
            pbar.set_postfix({"ms": f"{bt.total_ms:.0f}"})
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
                _clear_caches(loader_name)
            except Exception as e:
                print(f"  WARNING: cache clear failed: {e}", flush=True)

        batch_timings, system_samples = _run_epoch(loader, iter_fn, monitor, desc=f"{loader_name}/{run_type}")
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

def run_benchmark(config: BenchmarkConfig, result_dir: Path | None = None) -> None:
    loaders_to_run = config.loaders
    num_runs = config.num_runs
    dataset = config.dataset
    seed = config.seed

    torch.manual_seed(seed)

    if result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        result_dir = Path("exps/results") / dataset / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = Path(result_dir)
        timestamp = result_dir.name

    sha = _git_sha()
    cfg_dump = dataclasses.asdict(config)
    cfg_dump["gar_root"] = config.gar_root
    run_info = {
        "timestamp": timestamp,
        "git_sha": sha,
        "hardware": _hardware_info(),
        "notes": "",
        "config": cfg_dump,
    }
    (result_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Results dir: {result_dir}")

    for loader_name in loaders_to_run:
        print(f"\n=== {loader_name} ===", flush=True)
        loader = None
        try:
            if loader_name == "gar":
                loader = _make_gar_loader(config)
                iter_fn = _gar_iter
            elif loader_name in ("neo4j-global", "neo4j-per-node"):
                loader = _make_neo4j_loader(config, loader_name)
                iter_fn = _neo4j_iter
            elif loader_name == "pyg-inmem":
                loader = _make_pyg_loader(config)
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

    print(f"\nDone.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_PATH, help="Benchmark YAML.")
    p.add_argument("--result-dir", type=Path, default=None, help="Pre-created output directory.")
    args = p.parse_args()
    run_benchmark(load_config(args.config), result_dir=args.result_dir)


if __name__ == "__main__":
    main()
