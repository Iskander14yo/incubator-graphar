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
