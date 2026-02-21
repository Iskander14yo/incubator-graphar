from __future__ import annotations

import graphar as gar
import pytest
import torch
from torch_geometric.data import Data

from graphar.ml.torch import GARNeighborLoader


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
):
    if num_neighbors is None:
        num_neighbors = [5]
    return GARNeighborLoader(
        ldbc_graph,
        vertex_type="person",
        edge_type="knows",
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        batch_size=batch_size,
        shuffle=shuffle,
        features=features,
    )


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

    assert len(batch.n_id) == batch.x.size(0)
    assert tuple(batch.edge_index.shape[:1]) == (2,)
    if batch.edge_index.numel() > 0:
        assert int(batch.edge_index.max()) < batch.x.size(0)
    assert batch.edge_index.dtype == torch.long
    assert batch.n_id.dtype == torch.long
    assert isinstance(batch.x, torch.Tensor)
    assert batch.x.dtype in {torch.float16, torch.float32, torch.float64, torch.long, torch.int64}


def test_batch_size_is_respected_including_last_batch(ldbc_graph):
    loader = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3, 4], features=["id"], batch_size=2, shuffle=False
    )
    batches = list(iter(loader))

    assert [batch.batch_size for batch in batches] == [2, 2, 1]


def test_shuffle_false_is_deterministic_across_iterations(ldbc_graph):
    loader = _make_loader(
        ldbc_graph, input_nodes=[0, 1, 2, 3], features=["id"], batch_size=2, shuffle=False
    )
    first_pass_first_batch = next(iter(loader))
    second_pass_first_batch = next(iter(loader))

    assert first_pass_first_batch.input_id.tolist() == second_pass_first_batch.input_id.tolist()
    assert first_pass_first_batch.n_id[:2].tolist() == second_pass_first_batch.n_id[:2].tolist()


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
    assert batch.x.size(1) == expected_dim


def test_features_none_raises_on_non_numeric_properties(ldbc_graph):
    with pytest.raises(ValueError, match="Only numeric features are supported"):
        _make_loader(
            ldbc_graph, input_nodes=[0, 1], features=None, batch_size=2, shuffle=False
        )


def test_explicit_non_numeric_feature_raises(ldbc_graph):
    with pytest.raises(ValueError, match="Only numeric features are supported"):
        _make_loader(
            ldbc_graph,
            input_nodes=[0, 1],
            features=["id", "firstName"],
            batch_size=2,
            shuffle=False,
        )


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

    assert batch.batch_size == 2
    assert batch.n_id.tolist() == [0, 1]
    assert batch.edge_index.shape == (2, 0)
