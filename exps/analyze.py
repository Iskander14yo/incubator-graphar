#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LOADER_ORDER = ["gar", "neo4j-global", "neo4j-per-node", "pyg-inmem"]
RUN_TYPES = ["cold", "warm"]

_EXPS = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _mean(arr: list[float]) -> float:
    return float(np.mean(arr)) if arr else float("nan")


def _pct(arr: list[float], q: float) -> float:
    return float(np.percentile(arr, q)) if arr else float("nan")


def _peak(arr: list[float]) -> float:
    return float(max(arr)) if arr else float("nan")


def _fmt(v: float, spec: str = ".1f") -> str:
    return "n/a" if v != v else format(v, spec)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _is_run_dir(path: Path) -> bool:
    return (path / "run_info.json").exists()


def _find_latest_run(base: Path) -> Path:
    """Return the most recently modified run dir anywhere under base."""
    candidates = list(base.rglob("run_info.json"))
    if not candidates:
        sys.exit(f"No run directories found under {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def _resolve_path(args: argparse.Namespace) -> Path:
    if args.results is None:
        path = _find_latest_run(_EXPS / "results")
        print(f"Auto-selected: {path}")
        return path
    path = args.results
    if not path.exists():
        sys.exit(f"Directory not found: {path}")
    return path


def load_results(path: Path) -> dict[str, dict[str, dict]]:
    """Load loader JSONs from a run dir or all run subdirs of a parent dir.

    Returns:
        {loader: {run_type: {"batches": [...], "system_metrics": [...]}}}
    """
    run_dirs = [path] if _is_run_dir(path) else sorted(
        d for d in path.iterdir() if d.is_dir() and _is_run_dir(d)
    )
    if not run_dirs:
        sys.exit(f"No run directories found in {path}")

    agg: dict[str, dict[str, dict]] = {}
    for run_dir in run_dirs:
        for json_file in sorted(run_dir.glob("*.json")):
            if json_file.name == "run_info.json":
                continue
            loader = json_file.stem
            data = json.loads(json_file.read_text())
            agg.setdefault(loader, {})
            for run in data.get("runs", []):
                rt = run.get("type", "warm")
                bucket = agg[loader].setdefault(rt, {"batches": [], "system_metrics": []})
                bucket["batches"].extend(run.get("batches", []))
                bucket["system_metrics"].extend(run.get("system_metrics", []))
    return agg


def _non_profiled(batches: list[dict], loader: str) -> list[dict]:
    """Drop profiled batches for Neo4j (PROFILE adds real overhead)."""
    if loader.startswith("neo4j"):
        return [b for b in batches if not b.get("neo4j_profile")]
    return batches


def _ordered_loaders(agg: dict) -> list[str]:
    return [l for l in LOADER_ORDER if l in agg] + [
        l for l in sorted(agg) if l not in LOADER_ORDER
    ]


# ---------------------------------------------------------------------------
# Table data builders  (headers + rows, shared by ASCII and markdown output)
# ---------------------------------------------------------------------------

def _data_table1(agg: dict) -> tuple[list[str], list[list[str]]]:
    headers = ["Loader", "run", "mean (ms)", "P50 (ms)", "P95 (ms)", "Nodes", "Edges"]
    rows = []
    for loader in _ordered_loaders(agg):
        for rt in RUN_TYPES:
            if rt not in agg[loader]:
                continue
            batches = _non_profiled(agg[loader][rt]["batches"], loader)
            totals = [b["total_ms"] for b in batches]
            rows.append([
                loader, rt,
                _fmt(_mean(totals)),
                _fmt(_pct(totals, 50)),
                _fmt(_pct(totals, 95)),
                _fmt(_mean([b["sampled_nodes"] for b in batches]), ".0f"),
                _fmt(_mean([b["sampled_edges"] for b in batches]), ".0f"),
            ])
    return headers, rows


def _data_table2(agg: dict) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Loader", "run",
        "Retr mean", "Retr P50", "Retr P95",
        "Conv mean", "Conv P50", "Conv P95",
        "Samp mean", "Samp P50", "Samp P95",
        "Feat mean", "Feat P50", "Feat P95",
    ]
    rows = []
    stage_loaders = [l for l in _ordered_loaders(agg) if l == "gar" or l.startswith("neo4j")]
    for loader in stage_loaders:
        is_gar = loader == "gar"
        for rt in RUN_TYPES:
            if rt not in agg[loader]:
                continue
            batches = _non_profiled(agg[loader][rt]["batches"], loader)
            retr = [b["retrieval_ms"] for b in batches]
            conv = [b["conversion_ms"] for b in batches]
            samp = [b["sampling_ms"] for b in batches if b.get("sampling_ms") is not None] if is_gar else []
            feat = [b["feature_fetch_ms"] for b in batches if b.get("feature_fetch_ms") is not None] if is_gar else []

            def _triple(arr: list[float]) -> list[str]:
                if arr:
                    return [_fmt(_mean(arr)), _fmt(_pct(arr, 50)), _fmt(_pct(arr, 95))]
                return ["n/a", "n/a", "n/a"]

            rows.append([loader, rt] + _triple(retr) + _triple(conv) + _triple(samp) + _triple(feat))
    return headers, rows


def _data_table3(agg: dict) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Loader", "run",
        "CPU mean (%)", "CPU peak (%)",
        "RAM mean (MB)", "RAM peak (MB)",
        "Disk mean (MB/s)", "Disk peak (MB/s)",
    ]
    rows = []
    for loader in _ordered_loaders(agg):
        for rt in RUN_TYPES:
            if rt not in agg[loader]:
                continue
            metrics = agg[loader][rt]["system_metrics"]
            if not metrics:
                rows.append([loader, rt] + ["n/a"] * 6)
                continue
            cpu  = [m["cpu_pct"] for m in metrics]
            rss  = [m["rss_mb"] for m in metrics]
            disk = [m["disk_read_mb_s"] for m in metrics]
            rows.append([
                loader, rt,
                _fmt(_mean(cpu)), _fmt(_peak(cpu)),
                _fmt(_mean(rss)), _fmt(_peak(rss)),
                _fmt(_mean(disk)), _fmt(_peak(disk)),
            ])
    return headers, rows


def _data_neo4j_profile(agg: dict) -> tuple[list[str], list[list[str]]]:
    profiled = [
        b
        for loader in agg if loader.startswith("neo4j")
        for rt_data in agg[loader].values()
        for b in rt_data["batches"] if b.get("neo4j_profile")
    ]
    if not profiled:
        return [], []
    db_hits     = [b["neo4j_profile"].get("db_hits", 0) for b in profiled]
    cache_hits  = [b["neo4j_profile"].get("page_cache_hits", 0) for b in profiled]
    cache_miss  = [b["neo4j_profile"].get("page_cache_misses", 0) for b in profiled]
    op_times    = [b["neo4j_profile"].get("total_op_time_ms", 0) for b in profiled]
    headers = ["Stat", "DB Hits", "Cache Hits", "Cache Misses", "Total Op Time (ms)"]
    rows = [[
        f"mean (N={len(profiled)})",
        _fmt(_mean(db_hits), ".0f"),
        _fmt(_mean(cache_hits), ".0f"),
        _fmt(_mean(cache_miss), ".0f"),
        _fmt(_mean(op_times)),
    ]]
    return headers, rows


# ---------------------------------------------------------------------------
# Save table images
# ---------------------------------------------------------------------------

_TABLES = [
    ("table_1_main_comparison",       "Table 1: Main comparison",                 _data_table1),
    ("table_2_stage_breakdown",       "Table 2: Stage breakdown (GAR and Neo4j)", _data_table2),
    ("table_3_resource_usage",        "Table 3: Resource usage",                  _data_table3),
    ("table_4_neo4j_profile_summary", "Neo4j PROFILE summary",                    _data_neo4j_profile),
]



def _save_table_image(title: str, headers: list[str], rows: list[list[str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    n_cols = len(headers)
    n_rows = len(rows)

    # Figure size: scale with content
    char_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    total_chars = sum(char_widths)
    fig_w = max(6.0, min(total_chars * 0.13, 26.0))
    fig_h = max(1.2, n_rows * 0.38 + 1.0)
    font_size = 9 if n_cols <= 8 else 7

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_w = [cw / total_chars for cw in char_widths]
    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_w,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)

    # Header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#2C5F8A")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#1a3a5c")

    # Data rows — alternating background
    for i in range(1, n_rows + 1):
        bg = "#EBF3FB" if i % 2 == 0 else "#FFFFFF"
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#C8D8EA")

    ax.set_title(title, fontweight="bold", fontsize=font_size + 1, pad=10)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_table_images(agg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, title, builder in _TABLES:
        headers, rows = builder(agg)
        if rows:
            _save_table_image(title, headers, rows, out_dir / f"{fname}.png")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _best_run_type(agg: dict, loader: str) -> str:
    return "warm" if "warm" in agg[loader] else "cold"


def plot_results(agg: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = _ordered_loaders(agg)
    has_warm = any("warm" in agg[l] for l in loaders)
    run_label = "warm" if has_warm else "cold"

    # -- 1. Bar chart: mean batch time with std error bars -------------------
    means, stds, labels = [], [], []
    for loader in loaders:
        rt = _best_run_type(agg, loader)
        batches = _non_profiled(agg[loader][rt]["batches"], loader)
        totals = [b["total_ms"] for b in batches]
        means.append(_mean(totals))
        stds.append(float(np.std(totals)) if totals else 0.0)
        labels.append(loader)

    if means:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = list(range(len(labels)))
        ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Mean batch time (ms)")
        ax.set_title(f"Mean batch time — {run_label} runs")
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        p = out_dir / "01_mean_batch_time.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p}")

    # -- 2. Stacked bar: retrieval vs conversion (GAR + Neo4j) ---------------
    stage_loaders = [l for l in loaders if l == "gar" or l.startswith("neo4j")]
    if stage_loaders:
        retr_means, conv_means, s_labels = [], [], []
        for loader in stage_loaders:
            rt = _best_run_type(agg, loader)
            batches = _non_profiled(agg[loader][rt]["batches"], loader)
            retr_means.append(_mean([b["retrieval_ms"] for b in batches]))
            conv_means.append(_mean([b["conversion_ms"] for b in batches]))
            s_labels.append(loader)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = list(range(len(s_labels)))
        ax.bar(x, retr_means, label="Retrieval", color="steelblue", alpha=0.8)
        ax.bar(x, conv_means, bottom=retr_means, label="Conversion", color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(s_labels, rotation=15, ha="right")
        ax.set_ylabel("Mean time (ms)")
        ax.set_title(f"Stage breakdown — {run_label} runs")
        ax.legend()
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        p = out_dir / "02_stage_breakdown.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p}")

    # -- 3. Box plot: batch time distribution (no outliers) ------------------
    box_data, box_labels = [], []
    for loader in loaders:
        rt = _best_run_type(agg, loader)
        batches = _non_profiled(agg[loader][rt]["batches"], loader)
        box_data.append([b["total_ms"] for b in batches])
        box_labels.append(loader)

    if box_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showfliers=False)
        ax.set_ylabel("Batch time (ms)")
        ax.set_title(f"Batch time distribution — {run_label} runs (no outliers)")
        ax.grid(axis="y", alpha=0.4)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        p = out_dir / "03_batch_time_distribution.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "results", nargs="?", type=Path, default=None,
        help="Run directory or parent directory. Omit to auto-pick the latest run.",
    )
    ap.add_argument("--plots", action="store_true", help="Generate PNG plots.")
    args = ap.parse_args()

    results_path = _resolve_path(args)
    agg = load_results(results_path)
    if not agg:
        sys.exit("No loader results found.")

    out_dir = results_path / "plots"
    print(f"\nWriting output → {out_dir}")
    save_table_images(agg, out_dir)
    if args.plots:
        plot_results(agg, out_dir)


if __name__ == "__main__":
    main()
