import pyarrow as pa
import pytest

import graphar as gar
import graphar.ml as gar_ml


@pytest.fixture
def ldbc_graph(test_data_root):
    graph_path = f"{test_data_root}/ldbc_sample/parquet/ldbc_sample.graph.yml"
    return gar.GraphInfo.load(graph_path)


def test_sample_neighbors_basic(ldbc_graph):
    """Test basic neighbor sampling returns correct types."""
    seeds = [0, 1]
    fanout = [5]

    result = gar_ml.sample_neighbors(ldbc_graph, "person", "knows", seeds, fanout, seed=42)

    assert hasattr(result, "sampled_nodes")
    assert hasattr(result, "src_indices")
    assert hasattr(result, "dst_indices")

    assert len(result.src_indices) == len(result.dst_indices)
    # At least the seed nodes should be in result
    assert len(result.sampled_nodes) >= len(seeds)


def test_sample_neighbors_fanout_respected(ldbc_graph):
    """Test that fanout limit is respected."""
    seeds = [0]
    fanout = [2]

    result = gar_ml.sample_neighbors(ldbc_graph, "person", "knows", seeds, fanout, seed=42)

    # Number of neighbors for node 0 should be <= fanout[0]
    neighbors_of_seed = [
        dst for src, dst in zip(result.src_indices, result.dst_indices) if src == 0
    ]
    assert len(neighbors_of_seed) <= fanout[0]


def test_sample_neighbors_empty_seeds(ldbc_graph):
    """Test that empty seeds return empty result."""
    seeds = []
    fanout = [5]

    result = gar_ml.sample_neighbors(ldbc_graph, "person", "knows", seeds, fanout, seed=42)

    assert len(result.sampled_nodes) == 0
    assert len(result.src_indices) == 0
    assert len(result.dst_indices) == 0


def test_get_node_features_basic(ldbc_graph):
    """Test basic feature retrieval returns correct types."""
    node_ids = [0, 1, 2]
    properties = ["id"]

    table = gar_ml.get_node_features(ldbc_graph, "person", node_ids, properties)

    assert isinstance(table, pa.Table)
    assert table is not None
    assert table.num_rows == len(node_ids)
    assert table.num_columns == len(properties)
    assert table.column_names == properties


def test_get_node_features_multiple_props(ldbc_graph):
    """Test fetching multiple properties."""
    node_ids = [0, 1]
    properties = ["id", "firstName"]

    table = gar_ml.get_node_features(ldbc_graph, "person", node_ids, properties)

    assert table.num_rows == 2
    assert table.num_columns == 2
    assert set(table.column_names) == set(properties)


def test_get_node_features_empty_nodes(ldbc_graph):
    """Test fetching features for empty node list."""
    node_ids = []
    properties = ["id"]

    table = gar_ml.get_node_features(ldbc_graph, "person", node_ids, properties)

    assert table.num_rows == 0


def test_get_node_features_invalid_property(ldbc_graph):
    """Test that invalid property raises error."""
    node_ids = [0]
    properties = ["nonexistent_property"]

    with pytest.raises(ValueError, match="not found"):
        gar_ml.get_node_features(ldbc_graph, "person", node_ids, properties)


def test_sample_neighbors_invalid_edge_type(ldbc_graph):
    """Test that invalid edge type raises error."""
    seeds = [0]
    fanout = [5]

    with pytest.raises(ValueError, match="not found"):
        gar_ml.sample_neighbors(ldbc_graph, "person", "nonexistent_edge", seeds, fanout, seed=42)


def test_sample_neighbors_deterministic_with_seed(ldbc_graph):
    """Test that same seed produces same sampled result."""
    seeds = [0]
    fanout = [2]
    seed = 12345

    result1 = gar_ml.sample_neighbors(ldbc_graph, "person", "knows", seeds, fanout, seed=seed)
    result2 = gar_ml.sample_neighbors(ldbc_graph, "person", "knows", seeds, fanout, seed=seed)

    assert result1.sampled_nodes == result2.sampled_nodes
    assert result1.src_indices == result2.src_indices
    assert result1.dst_indices == result2.dst_indices


def test_feature_cache_basic(ldbc_graph):
    """Test FeatureCache hit/miss counting and stats API."""
    cache = gar_ml.FeatureCache(64 * 1024 * 1024)
    assert cache.hits == 0
    assert cache.misses == 0
    assert cache.hit_rate == 0.0
    assert cache.num_nodes == 0
    assert cache.size_mb == pytest.approx(0.0)
    assert cache.max_size_mb == pytest.approx(64.0)

    # First call — misses (chunks read from disk)
    gar_ml.get_node_features(ldbc_graph, "person", [0, 1, 2], ["id"], cache=cache)
    assert cache.misses > 0
    assert cache.hits == 0
    assert cache.num_nodes > 0

    misses_after_first = cache.misses

    # Second identical call — chunks now in cache
    gar_ml.get_node_features(ldbc_graph, "person", [0, 1, 2], ["id"], cache=cache)
    assert cache.hits > 0
    assert cache.misses == misses_after_first  # no new misses
    assert cache.hit_rate > 0.0


def test_feature_cache_returns_same_results(ldbc_graph):
    """Cached and uncached get_node_features must return identical data."""
    cache = gar_ml.FeatureCache(64 * 1024 * 1024)
    node_ids = [0, 1, 2, 5, 10]
    props = ["id", "firstName"]

    no_cache = gar_ml.get_node_features(ldbc_graph, "person", node_ids, props)
    first_cached = gar_ml.get_node_features(ldbc_graph, "person", node_ids, props, cache=cache)
    second_cached = gar_ml.get_node_features(ldbc_graph, "person", node_ids, props, cache=cache)

    assert no_cache.to_pydict() == first_cached.to_pydict()
    assert no_cache.to_pydict() == second_cached.to_pydict()


def test_static_cache_matches_uncached_and_records_hits(ldbc_graph):
    sel = gar_ml.DegreeHotNodeSelector(ldbc_graph, "person", "knows")
    top = sel.select(32)
    cache = gar_ml.StaticFeatureCache(ldbc_graph)
    cache.pin("person", top, ["id"])

    # Include ids from `top` so lookups are not all misses (low-degree fixtures may not overlap [0,7,...]).
    node_ids = [top[0], top[1], 0, 7, 2]
    plain = gar_ml.get_node_features(ldbc_graph, "person", node_ids, ["id"])
    got = gar_ml.get_node_features(
        ldbc_graph, "person", node_ids, ["id"], static_cache=cache
    )
    assert plain.to_pydict() == got.to_pydict()
    assert cache.hits > 0


def test_get_node_features_cache_arguments_mutex(ldbc_graph):
    fc = gar_ml.FeatureCache(1024 * 1024)
    sc = gar_ml.StaticFeatureCache(ldbc_graph)
    with pytest.raises(RuntimeError, match="only one"):
        gar_ml.get_node_features(
            ldbc_graph, "person", [0], ["id"], cache=fc, static_cache=sc
        )


def test_feature_cache_clear(ldbc_graph):
    """Clear empties entries but preserves cumulative hit/miss stats."""
    cache = gar_ml.FeatureCache(64 * 1024 * 1024)
    gar_ml.get_node_features(ldbc_graph, "person", [0], ["id"], cache=cache)
    gar_ml.get_node_features(ldbc_graph, "person", [0], ["id"], cache=cache)

    misses = cache.misses
    hits = cache.hits
    cache.clear()

    assert cache.num_nodes == 0
    assert cache.size_mb == pytest.approx(0.0)
    assert cache.misses == misses  # stats preserved
    assert cache.hits == hits


def test_sample_neighbors_preserves_sampling_order(ldbc_graph):
    seeds = [0, 1]
    fanout = [10]

    result = gar_ml.sample_neighbors(ldbc_graph, "person", "knows", seeds, fanout, seed=42)

    assert result.sampled_nodes == [0, 1, 87, 623, 849, 58, 318, 538, 539, 696]
    assert list(zip(result.src_indices, result.dst_indices)) == [
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
    ]
