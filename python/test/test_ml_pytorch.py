from __future__ import annotations

import threading
import unittest.mock as mock

import pyarrow as pa
import pytest
import torch
from torch_geometric.data import Data

import graphar as gar
import graphar.ml.torch as gar_torch
from graphar.ml.torch import BatchProfile, GARNeighborLoader


@pytest.fixture
def ldbc_graph(test_data_root):
    graph_path = f"{test_data_root}/ldbc_sample/parquet/ldbc_sample.graph.yml"
    return gar.GraphInfo.load(graph_path)


def _make_loader(
    ldbc_graph,
    *,
    input_nodes=None,
    num_neighbors=None,
    batch_size=2,
    shuffle=False,
    features=None,
    num_workers=0,
    prefetch_windows=0,
    feature_buffer_batches=1,
    edge_ram_for_loader_mb=0,
    feature_ram_for_loader_mb=0,
):
    if num_neighbors is None:
        num_neighbors = [5]
    kwargs = dict(
        graph_info=ldbc_graph,
        vertex_type="person",
        edge_type="knows",
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        batch_size=batch_size,
        shuffle=shuffle,
        features=features,
        num_workers=num_workers,
        prefetch_windows=prefetch_windows,
        feature_buffer_batches=feature_buffer_batches,
        edge_ram_for_loader_mb=edge_ram_for_loader_mb,
        feature_ram_for_loader_mb=feature_ram_for_loader_mb,
    )
    return GARNeighborLoader(**kwargs)


def _batch_tensors(batch: Data) -> tuple[torch.Tensor, torch.Tensor]:
    assert batch.x is not None
    assert batch.edge_index is not None
    return batch.x, batch.edge_index


class _FakeSampling:
    def __init__(
        self,
        sampled_nodes,
        src_indices,
        dst_indices,
        num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop,
    ):
        self.sampled_nodes = sampled_nodes
        self.src_indices = src_indices
        self.dst_indices = dst_indices
        self.num_sampled_nodes_per_hop = num_sampled_nodes_per_hop
        self.num_sampled_edges_per_hop = num_sampled_edges_per_hop


def _make_mock_loader(
    *,
    input_nodes,
    batch_size=2,
    num_workers=0,
    prefetch_windows=0,
    feature_buffer_batches=1,
):
    return GARNeighborLoader(
        graph_info=object(),
        vertex_type="person",
        edge_type="knows",
        num_neighbors=[2],
        input_nodes=input_nodes,
        batch_size=batch_size,
        shuffle=False,
        features=["feat"],
        num_workers=num_workers,
        prefetch_windows=prefetch_windows,
        feature_buffer_batches=feature_buffer_batches,
    )


def _mock_sampling(seed_nodes, *_args, **_kwargs):
    if seed_nodes == [0, 1]:
        return _FakeSampling([0, 1, 10], [0, 1], [2, 2], [2, 1], [2])
    if seed_nodes == [2, 3]:
        return _FakeSampling([2, 3, 10, 11], [0, 1, 1], [2, 2, 3], [2, 2], [3])
    if seed_nodes == [4]:
        return _FakeSampling([4, 12], [0], [1], [1, 1], [1])
    raise AssertionError(f"unexpected seed nodes: {seed_nodes}")


def _mock_feature_table(_graph_info, _vertex_type, node_ids, _features, chunk_manager=None):
    del chunk_manager
    return pa.table({"feat": pa.array(node_ids, type=pa.int64())})


def test_loader_yields_data_with_expected_contract(ldbc_graph):
    loader = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2, shuffle=False
    )
    batch = next(iter(loader))

    assert isinstance(batch, Data)
    assert hasattr(batch, "x")
    assert hasattr(batch, "edge_index")
    assert hasattr(batch, "batch_size")
    assert hasattr(batch, "n_id")
    assert hasattr(batch, "input_id")
    assert hasattr(batch, "num_sampled_nodes")
    assert hasattr(batch, "num_sampled_edges")
    assert hasattr(batch, "vertex_type")
    assert hasattr(batch, "edge_type")


def test_seeds_are_first_and_match_input_id(ldbc_graph):
    loader = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2, shuffle=False
    )
    batch = next(iter(loader))

    assert batch.batch_size == 2
    assert batch.input_id.tolist() == [0, 1]
    assert batch.n_id[: batch.batch_size].tolist() == batch.input_id.tolist()


def test_batch_shape_index_consistency_and_dtypes(ldbc_graph):
    loader = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2, shuffle=False
    )
    batch = next(iter(loader))
    x, edge_index = _batch_tensors(batch)

    assert len(batch.n_id) == x.size(0)
    assert tuple(edge_index.shape[:1]) == (2,)
    if edge_index.numel() > 0:
        assert int(edge_index.max()) < x.size(0)
    assert edge_index.dtype == torch.long
    assert batch.n_id.dtype == torch.long
    assert isinstance(x, torch.Tensor)
    assert x.dtype in {torch.float16, torch.float32, torch.float64, torch.long, torch.int64}


def test_batch_size_is_respected_including_last_batch(ldbc_graph):
    loader = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3, 4], features=["id"], batch_size=2, shuffle=False
    )
    batches = list(iter(loader))

    assert [batch.batch_size for batch in batches] == [2, 2, 1]


def _epoch_signature(loader):
    return [
        (
            batch.input_id.tolist(),
            batch.n_id.tolist(),
            batch.edge_index.tolist(),
        )
        for batch in loader
    ]


def test_rng_stream_is_deterministic_across_sessions_shuffle_false(ldbc_graph):
    loader1 = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2, shuffle=False
    )
    loader2 = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2, shuffle=False
    )

    # Compare two epochs to ensure RNG progression is reproducible, not reset per epoch.
    assert _epoch_signature(loader1) == _epoch_signature(loader2)
    assert _epoch_signature(loader1) == _epoch_signature(loader2)


def test_rng_stream_is_deterministic_across_sessions_shuffle_true(ldbc_graph):
    loader1 = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2, shuffle=True
    )
    loader2 = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2, shuffle=True
    )
    assert _epoch_signature(loader1) == _epoch_signature(loader2)


@pytest.mark.parametrize(
    ("input_nodes", "expected_first_seeds"),
    [
        ([0, 2, 4], [0, 2]),
        (torch.tensor([0, 2, 4]), [0, 2]),
        ([True, False, True, False, True], [0, 2]),
        (torch.tensor([True, False, True, False, True]), [0, 2]),
    ],
)
def test_input_nodes_formats(ldbc_graph, input_nodes, expected_first_seeds):
    loader = _make_loader(
        ldbc_graph, input_nodes=input_nodes, features=["id"], batch_size=2, shuffle=False
    )
    batch = next(iter(loader))
    assert batch.input_id.tolist() == expected_first_seeds


def test_input_nodes_none_uses_all_nodes(ldbc_graph):
    loader = _make_loader(ldbc_graph, input_nodes=None, features=["id"], batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert batch.batch_size == 4
    assert batch.input_id.tolist() == [0, 1, 2, 3]


def test_input_nodes_none_shuffle_false_is_lazy(ldbc_graph):
    with mock.patch.object(
        gar_torch,
        "_normalize_input_nodes",
        side_effect=AssertionError("lazy all-node path should bypass normalization"),
    ):
        loader = _make_loader(
            ldbc_graph,
            input_nodes=None,
            features=["id"],
            batch_size=4,
            shuffle=False,
        )

    assert loader._input_nodes is None
    assert loader._input_node_count == ldbc_graph.get_vertex_count("person")


def test_empty_input_nodes_yields_no_batches(ldbc_graph):
    loader = _make_loader(ldbc_graph, input_nodes=[], features=["id"], batch_size=2, shuffle=False)
    assert list(iter(loader)) == []


@pytest.mark.parametrize(
    ("features", "expected_dim"),
    [
        ([], 0),
        (["id"], 1),
    ],
)
def test_feature_selection_sanity(ldbc_graph, features, expected_dim):
    loader = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2], features=features, batch_size=2, shuffle=False
    )
    batch = next(iter(loader))
    x, _ = _batch_tensors(batch)
    assert x.size(1) == expected_dim


def test_chunk_manager_is_owned_by_loader(ldbc_graph):
    loader = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2, shuffle=False
    )
    list(loader)

    stats = loader.chunk_manager_stats()
    assert stats["requests"] > 0
    assert stats["leaders"] == stats["requests"]
    assert stats["completed"] == stats["requests"]
    assert stats["failed"] == 0
    assert stats["ram_cache_hits"] == 0
    assert stats["ram_cache_misses"] == 0
    assert stats["ram_cache_evictions"] == 0
    assert stats["ram_cache_bytes"] == 0


def test_chunk_manager_uses_ram_budget(ldbc_graph):
    loader = _make_loader(
        ldbc_graph,
        input_nodes=[0, 1, 2, 3],
        features=["id"],
        batch_size=2,
        shuffle=False,
        edge_ram_for_loader_mb=1,
    )
    list(loader)

    stats = loader.chunk_manager_stats()
    assert stats["requests"] > 0
    assert stats["ram_cache_misses"] > 0
    assert stats["ram_cache_bytes"] > 0


def test_feature_chunk_manager_uses_separate_ram_budget(ldbc_graph):
    loader = _make_loader(
        ldbc_graph,
        input_nodes=[0, 1, 2, 3],
        features=["id"],
        batch_size=2,
        shuffle=False,
        edge_ram_for_loader_mb=0,
        feature_ram_for_loader_mb=1,
    )
    list(loader)

    sampling_stats = loader.chunk_manager_stats()
    feature_stats = loader.feature_chunk_manager_stats()
    assert sampling_stats["ram_cache_bytes"] == 0
    assert feature_stats["requests"] > 0
    assert feature_stats["ram_cache_misses"] > 0
    assert feature_stats["ram_cache_bytes"] > 0


def test_feature_chunk_manager_cache_can_be_disabled(ldbc_graph):
    loader = _make_loader(
        ldbc_graph,
        input_nodes=[0, 1, 2, 3],
        features=["id"],
        batch_size=2,
        shuffle=False,
        edge_ram_for_loader_mb=1,
        feature_ram_for_loader_mb=0,
    )
    list(loader)

    sampling_stats = loader.chunk_manager_stats()
    feature_stats = loader.feature_chunk_manager_stats()
    assert sampling_stats["ram_cache_bytes"] > 0
    assert feature_stats["requests"] > 0
    assert feature_stats["ram_cache_hits"] == 0
    assert feature_stats["ram_cache_misses"] == 0
    assert feature_stats["ram_cache_bytes"] == 0


def test_chunk_manager_is_used_for_sampling_without_features(ldbc_graph):
    loader = _make_loader(
        ldbc_graph,
        input_nodes=[0, 1, 2, 3],
        features=[],
        batch_size=2,
        shuffle=False,
    )
    list(loader)

    stats = loader.chunk_manager_stats()
    assert stats["requests"] > 0
    assert stats["completed"] == stats["requests"]
    assert stats["failed"] == 0


def test_negative_edge_ram_for_loader_is_rejected(ldbc_graph):
    with pytest.raises(ValueError, match="edge_ram_for_loader_mb"):
        _make_loader(ldbc_graph, edge_ram_for_loader_mb=-1)


def test_negative_prefetch_windows_is_rejected(ldbc_graph):
    with pytest.raises(ValueError, match="prefetch_windows"):
        _make_loader(ldbc_graph, prefetch_windows=-1)


def test_negative_feature_buffer_batches_is_rejected(ldbc_graph):
    with pytest.raises(ValueError, match="feature_buffer_batches"):
        _make_loader(ldbc_graph, feature_buffer_batches=0)


def test_negative_feature_ram_for_loader_is_rejected(ldbc_graph):
    with pytest.raises(ValueError, match="feature_ram_for_loader_mb"):
        _make_loader(ldbc_graph, feature_ram_for_loader_mb=-1)


def test_valid_batch_with_zero_fanout(ldbc_graph):
    loader = _make_loader(
        ldbc_graph,
        input_nodes=[0, 1],
        num_neighbors=[0],
        features=["id"],
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))
    _, edge_index = _batch_tensors(batch)

    assert batch.batch_size == 2
    assert batch.n_id.tolist() == [0, 1]
    assert edge_index.shape == (2, 0)
    assert batch.num_sampled_nodes.tolist() == [2, 0]
    assert batch.num_sampled_edges.tolist() == [0]


def test_loader_keeps_sampler_order(ldbc_graph):
    loader = _make_loader(
        ldbc_graph,
        input_nodes=[0, 1],
        num_neighbors=[10],
        features=["id"],
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))

    assert batch.n_id.tolist() == [0, 1, 87, 623, 849, 58, 318, 538, 539, 696]
    assert batch.edge_index.tolist() == [
        [0, 0, 0, 1, 1, 1, 1, 1],
        [2, 3, 4, 5, 6, 7, 8, 9],
    ]
    assert batch.num_sampled_nodes.tolist() == [2, 8]
    assert batch.num_sampled_edges.tolist() == [8]


def test_multi_worker_matches_single_worker_shuffle_false(ldbc_graph):
    single_worker_loader = _make_loader(
        ldbc_graph,
        input_nodes=[0, 1, 2, 3, 4, 5],
        features=["id"],
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    multi_worker_loader = _make_loader(
        ldbc_graph,
        input_nodes=[0, 1, 2, 3, 4, 5],
        features=["id"],
        batch_size=2,
        shuffle=False,
        num_workers=2,
    )

    assert _epoch_signature(single_worker_loader) == _epoch_signature(multi_worker_loader)


def test_multi_worker_shuffle_false_keeps_batch_order(ldbc_graph):
    loader = _make_loader(
        ldbc_graph,
        input_nodes=[0, 1, 2, 3, 4, 5],
        features=["id"],
        batch_size=2,
        shuffle=False,
        num_workers=2,
        feature_buffer_batches=2,
        prefetch_windows=2,
    )

    assert [batch.input_id.tolist() for batch in loader] == [[0, 1], [2, 3], [4, 5]]


def test_multi_worker_reorders_out_of_order_windows():
    loader = _make_mock_loader(
        input_nodes=list(range(8)),
        num_workers=2,
        prefetch_windows=2,
        feature_buffer_batches=2,
    )
    release_first = threading.Event()
    second_window_started = threading.Event()
    seen_input_ids = []
    thread_error = []

    def tracked_build(window_jobs):
        first_seed = window_jobs[0][0][0]
        if first_seed == 0:
            if not release_first.wait(timeout=5.0):
                raise TimeoutError("timed out waiting to release first window")
        else:
            second_window_started.set()
        pairs = []
        for seed_nodes, _seed in window_jobs:
            batch = Data(
                x=torch.empty((len(seed_nodes), 0), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
            )
            batch.batch_size = len(seed_nodes)
            batch.n_id = torch.tensor(seed_nodes, dtype=torch.long)
            batch.input_id = torch.tensor(seed_nodes, dtype=torch.long)
            batch.num_sampled_nodes = torch.tensor([len(seed_nodes)], dtype=torch.long)
            batch.num_sampled_edges = torch.empty((0,), dtype=torch.long)
            batch.vertex_type = loader.vertex_type
            batch.edge_type = loader.edge_type
            pairs.append((batch, BatchProfile(0.0, 0.0, 0.0, 0.0)))
        return pairs

    def consume():
        try:
            seen_input_ids.extend(batch.input_id.tolist() for batch in loader)
        except Exception as exc:  # pragma: no cover - surfaced by assertion below
            thread_error.append(exc)

    with mock.patch.object(loader, "_build_window", side_effect=tracked_build):
        consumer = threading.Thread(target=consume)
        consumer.start()
        assert second_window_started.wait(timeout=5.0)
        release_first.set()
        consumer.join(timeout=10.0)

    assert not consumer.is_alive()
    assert thread_error == []
    assert seen_input_ids == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_multi_worker_is_deterministic_across_sessions(ldbc_graph):
    loader1 = _make_loader(
        ldbc_graph,
        input_nodes=[0, 1, 2, 3, 4, 5],
        features=["id"],
        batch_size=2,
        shuffle=True,
        num_workers=2,
    )
    loader2 = _make_loader(
        ldbc_graph,
        input_nodes=[0, 1, 2, 3, 4, 5],
        features=["id"],
        batch_size=2,
        shuffle=True,
        num_workers=2,
    )

    assert _epoch_signature(loader1) == _epoch_signature(loader2)
    assert _epoch_signature(loader1) == _epoch_signature(loader2)


# ---------------------------------------------------------------------------
# profile()
# ---------------------------------------------------------------------------

def test_profile_yields_batch_profile_pairs(ldbc_graph):
    loader = _make_loader(ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2)
    for batch, prof in loader.profile():
        assert isinstance(batch, Data)
        assert isinstance(prof, BatchProfile)


def test_profile_timings_are_non_negative(ldbc_graph):
    loader = _make_loader(ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2)
    for _, prof in loader.profile():
        assert prof.total_ms >= 0
        assert prof.sampling_ms >= 0
        assert prof.feature_fetch_ms >= 0
        assert prof.conversion_ms >= 0


def test_profile_total_covers_stages(ldbc_graph):
    loader = _make_loader(ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2)
    for _, prof in loader.profile():
        assert prof.total_ms >= prof.sampling_ms + prof.feature_fetch_ms + prof.conversion_ms


def test_profile_batches_match_iter(ldbc_graph):
    """profile() must yield the same batches as __iter__."""
    iter_loader = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3, 4, 5], features=["id"], batch_size=2, shuffle=False
    )
    prof_loader = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3, 4, 5], features=["id"], batch_size=2, shuffle=False
    )

    iter_sig = _epoch_signature(iter_loader)
    prof_sig = [
        (batch.input_id.tolist(), batch.n_id.tolist(), batch.edge_index.tolist())
        for batch, _ in prof_loader.profile()
    ]
    assert iter_sig == prof_sig


def test_profile_works_with_multi_worker(ldbc_graph):
    loader = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3, 4, 5], features=["id"], batch_size=2, num_workers=2
    )
    pairs = list(loader.profile())
    assert all(isinstance(prof, BatchProfile) for _, prof in pairs)
    assert all(prof.total_ms >= 0 for _, prof in pairs)


def test_feature_buffer_batches_one_keeps_per_batch_fetching():
    loader = _make_mock_loader(input_nodes=[0, 1, 2, 3, 4], feature_buffer_batches=1)

    with (
        mock.patch.object(gar_torch.gar_ml, "sample_neighbors", side_effect=_mock_sampling),
        mock.patch.object(gar_torch.gar_ml, "get_node_features", side_effect=_mock_feature_table) as feature_mock,
    ):
        batches = list(loader)

    assert len(batches) == 3
    assert feature_mock.call_count == 3
    assert [call.args[2] for call in feature_mock.call_args_list] == [
        [0, 1, 10],
        [2, 3, 10, 11],
        [4, 12],
    ]
    assert loader.feature_window_stats()["windows"] == 3


def test_feature_buffered_window_fetches_once_and_preserves_batch_payload():
    loader = _make_mock_loader(input_nodes=[0, 1, 2, 3], feature_buffer_batches=2)

    with (
        mock.patch.object(gar_torch.gar_ml, "sample_neighbors", side_effect=_mock_sampling),
        mock.patch.object(gar_torch.gar_ml, "get_node_features", side_effect=_mock_feature_table) as feature_mock,
    ):
        pairs = loader._build_window([([0, 1], 1), ([2, 3], 2)])

    assert feature_mock.call_count == 1
    assert feature_mock.call_args.args[2] == [0, 1, 10, 2, 3, 11]

    first_batch, first_prof = pairs[0]
    second_batch, second_prof = pairs[1]
    assert first_batch.input_id.tolist() == [0, 1]
    assert first_batch.n_id.tolist() == [0, 1, 10]
    assert first_batch.x.tolist() == [[0.0], [1.0], [10.0]]
    assert first_batch.edge_index.tolist() == [[0, 1], [2, 2]]
    assert second_batch.input_id.tolist() == [2, 3]
    assert second_batch.n_id.tolist() == [2, 3, 10, 11]
    assert second_batch.x.tolist() == [[2.0], [3.0], [10.0], [11.0]]
    assert second_batch.edge_index.tolist() == [[0, 1, 1], [2, 2, 3]]
    assert first_prof.feature_fetch_ms == pytest.approx(second_prof.feature_fetch_ms)
    assert first_prof.conversion_ms == pytest.approx(second_prof.conversion_ms)

    stats = loader.feature_window_stats()
    assert stats["windows"] == 1
    assert stats["window_batches_sum"] == 2
    assert stats["window_batches_max"] == 2
    assert stats["window_nodes_sum"] == 7
    assert stats["window_unique_nodes_sum"] == 6
    assert stats["window_reuse_ratio"] == pytest.approx(7 / 6)


def test_feature_window_profiles_amortize_shared_timings():
    loader = _make_mock_loader(input_nodes=[0, 1, 2, 3], feature_buffer_batches=2)
    perf_samples = iter([0.0, 0.01, 0.01, 0.03, 0.03, 0.07, 0.07, 0.09])

    with (
        mock.patch.object(gar_torch.gar_ml, "sample_neighbors", side_effect=_mock_sampling),
        mock.patch.object(gar_torch.gar_ml, "get_node_features", side_effect=_mock_feature_table),
        mock.patch.object(gar_torch.time, "perf_counter", side_effect=lambda: next(perf_samples)),
    ):
        pairs = loader._build_window([([0, 1], 1), ([2, 3], 2)])

    profiles = [prof for _, prof in pairs]
    assert profiles[0].sampling_ms == pytest.approx(10.0)
    assert profiles[1].sampling_ms == pytest.approx(20.0)
    assert profiles[0].feature_fetch_ms == pytest.approx(20.0)
    assert profiles[1].feature_fetch_ms == pytest.approx(20.0)
    assert profiles[0].conversion_ms == pytest.approx(10.0)
    assert profiles[1].conversion_ms == pytest.approx(10.0)
    assert profiles[0].total_ms == pytest.approx(40.0)
    assert profiles[1].total_ms == pytest.approx(50.0)


def test_feature_buffered_profile_handles_last_partial_window():
    loader = _make_mock_loader(input_nodes=[0, 1, 2, 3, 4], feature_buffer_batches=2)

    with (
        mock.patch.object(gar_torch.gar_ml, "sample_neighbors", side_effect=_mock_sampling),
        mock.patch.object(gar_torch.gar_ml, "get_node_features", side_effect=_mock_feature_table) as feature_mock,
    ):
        pairs = list(loader.profile())

    assert [batch.input_id.tolist() for batch, _ in pairs] == [[0, 1], [2, 3], [4]]
    assert [call.args[2] for call in feature_mock.call_args_list] == [
        [0, 1, 10, 2, 3, 11],
        [4, 12],
    ]

    stats = loader.feature_window_stats()
    assert stats["windows"] == 2
    assert stats["window_batches_sum"] == 3
    assert stats["window_batches_mean"] == pytest.approx(1.5)
    assert stats["window_batches_max"] == 2


# ---------------------------------------------------------------------------
# bounded prefetch
# ---------------------------------------------------------------------------

def test_bounded_prefetch_limits_concurrency(ldbc_graph):
    """At most num_workers _build_window calls should run simultaneously."""
    num_workers = 2
    loader = _make_loader(
        ldbc_graph,
        input_nodes=list(range(8)),
        features=["id"],
        batch_size=2,
        shuffle=False,
        num_workers=num_workers,
    )

    peak = 0
    active = 0
    lock = threading.Lock()
    real_build = loader._build_window

    def tracked_build(window_jobs):
        nonlocal peak, active
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return real_build(window_jobs)
        finally:
            with lock:
                active -= 1

    with mock.patch.object(loader, "_build_window", side_effect=tracked_build):
        list(loader)

    assert peak <= num_workers


def test_prefetch_windows_submit_past_slow_head(ldbc_graph):
    loader = _make_loader(
        ldbc_graph,
        input_nodes=list(range(16)),
        features=["id"],
        batch_size=2,
        shuffle=False,
        num_workers=2,
        prefetch_windows=4,
        feature_buffer_batches=2,
    )

    real_build = loader._build_window
    release_first = threading.Event()
    started_fourth = threading.Event()
    started_windows = set()
    started_lock = threading.Lock()
    thread_error = []

    def tracked_build(window_jobs):
        window_idx = window_jobs[0][0][0] // 4
        with started_lock:
            started_windows.add(window_idx)
            if window_idx == 3:
                started_fourth.set()
        if window_idx == 0:
            if not release_first.wait(timeout=5.0):
                raise TimeoutError("timed out waiting to release first window")
        return real_build(window_jobs)

    def consume():
        try:
            list(loader)
        except Exception as exc:  # pragma: no cover - surfaced by assertion below
            thread_error.append(exc)

    with mock.patch.object(loader, "_build_window", side_effect=tracked_build):
        consumer = threading.Thread(target=consume)
        consumer.start()
        assert started_fourth.wait(timeout=5.0)
        with started_lock:
            assert started_windows >= {0, 1, 2, 3}
        release_first.set()
        consumer.join(timeout=10.0)

    assert not consumer.is_alive()
    assert thread_error == []
