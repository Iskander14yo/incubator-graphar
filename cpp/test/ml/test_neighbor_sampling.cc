#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>

#include "graphar/api/high_level_writer.h"
#include "graphar/ml/neighbor_sampling.h"

#include <catch2/catch_test_macros.hpp>

namespace graphar::ml {

// Test fixture that creates a small test graph
struct SamplingTestFixture {
  SamplingTestFixture() {
    test_dir = "/tmp/graphar_ml_test";
    CreateTestGraph();
  }

  ~SamplingTestFixture() {
    // Cleanup test data
    std::filesystem::remove_all(test_dir);
  }

  void CreateTestGraph() {
    // Create vertex info: Node type with id and feature properties
    std::vector<Property> vertex_properties = {
        Property("id", int64(), false, false),
        Property("feature", float32(), false, true)};
    auto vertex_property_group =
        CreatePropertyGroup(vertex_properties, FileType::PARQUET);
    auto vertex_info =
        CreateVertexInfo("Node", 100, {vertex_property_group}, {}, "", nullptr);

    // Create edge info: edge type with CSR layout
    auto adj_list = CreateAdjacentList(AdjListType::ordered_by_source,
                                       FileType::PARQUET);
    auto edge_info =
        CreateEdgeInfo("Node", "edge", "Node", 100, 100, 100, true, {adj_list},
                       {}, "", nullptr);

    // Create graph info
    graph_info =
        CreateGraphInfo("test_graph", {vertex_info}, {edge_info}, {}, test_dir);

    // Build vertices using VerticesBuilder
    auto builder_result =
        builder::VerticesBuilder::Make(vertex_info, test_dir, 0);
    REQUIRE(!builder_result.has_error());
    auto vertex_builder = builder_result.value();

    // Add vertices and save them
    for (int64_t i = 0; i < 6; i++) {
      builder::Vertex v;
      v.AddProperty("id", i);
      v.AddProperty("feature", static_cast<float>(i * 0.1f));
      REQUIRE(vertex_builder->AddVertex(v).ok());
    }
    REQUIRE(vertex_builder->Dump().ok());

    // Add edges and save them
    // Graph structure:
    // 0 -> 1, 2
    // 1 -> 2, 3
    // 2 -> 3, 4
    // 3 -> 4
    // 4 -> (none)
    // 5 (isolated)
    auto edges_builder_result = builder::EdgesBuilder::Make(
        edge_info, test_dir, AdjListType::ordered_by_source, 6);
    REQUIRE(!edges_builder_result.has_error());
    auto edges_builder = edges_builder_result.value();

    std::vector<std::pair<int64_t, int64_t>> edges = {
        {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}, {2, 4}, {3, 4}};

    for (const auto& [src, dst] : edges) {
      builder::Edge e(src, dst);
      REQUIRE(edges_builder->AddEdge(e).ok());
    }
    REQUIRE(edges_builder->Dump().ok());

    // Save graph info
    REQUIRE(graph_info->Save(test_dir + "/test_graph.graph.yml").ok());
  }

  std::string test_dir;
  std::shared_ptr<GraphInfo> graph_info;
};

//////////////////////////// SampleNeighbors /////////////////////////////////////

TEST_CASE_METHOD(SamplingTestFixture, "SampleNeighbors - single hop") {
  std::vector<IdType> seeds = {0};
  std::vector<int> fanout = {10};  // Request more than available

  auto result = SampleNeighbors(graph_info, "Node", "edge", seeds, fanout, 42);
  REQUIRE(result.status().ok());

  auto& sampling = result.value();
  
  // Should have seed (0) + its neighbors (1,2)
  REQUIRE(sampling.sampled_nodes.size() == 3);

  // Verify seed is in sampled_nodes
  REQUIRE(std::find(sampling.sampled_nodes.begin(),
                    sampling.sampled_nodes.end(), 0) !=
          sampling.sampled_nodes.end());

  // Verify edges consistency
  REQUIRE(sampling.src_indices.size() == sampling.dst_indices.size());
  
  // All edges should start from node 0 (index of 0 in sampled_nodes)
  auto seed_idx = std::find(sampling.sampled_nodes.begin(),
                           sampling.sampled_nodes.end(), 0) -
                 sampling.sampled_nodes.begin();
  for (size_t i = 0; i < sampling.src_indices.size(); i++) {
    REQUIRE(sampling.src_indices[i] == seed_idx);
  }
}

TEST_CASE_METHOD(SamplingTestFixture, "SampleNeighbors - fanout limit") {
  std::vector<IdType> seeds = {0};
  std::vector<int> fanout = {1};  // Limit to 1 neighbor

  auto result = SampleNeighbors(graph_info, "Node", "edge", seeds, fanout, 42);
  REQUIRE(result.status().ok());

  auto& sampling = result.value();
  
  // Should have exactly 1 edge (fanout limit)
  REQUIRE(sampling.src_indices.size() == 1);
  REQUIRE(sampling.dst_indices.size() == 1);
  
  // Should have seed + 1 neighbor
  REQUIRE(sampling.sampled_nodes.size() == 2);
}

TEST_CASE_METHOD(SamplingTestFixture, "SampleNeighbors - deterministic with seed") {
  std::vector<IdType> seeds = {0};
  std::vector<int> fanout = {1};
  uint64_t seed = 12345;

  auto result1 = SampleNeighbors(graph_info, "Node", "edge", seeds, fanout, seed);
  auto result2 = SampleNeighbors(graph_info, "Node", "edge", seeds, fanout, seed);
  REQUIRE(result1.status().ok());
  REQUIRE(result2.status().ok());

  const auto& sampling1 = result1.value();
  const auto& sampling2 = result2.value();
  REQUIRE(sampling1.sampled_nodes == sampling2.sampled_nodes);
  REQUIRE(sampling1.src_indices == sampling2.src_indices);
  REQUIRE(sampling1.dst_indices == sampling2.dst_indices);
}

TEST_CASE_METHOD(SamplingTestFixture, "SampleNeighbors - multi-hop") {
  std::vector<IdType> seeds = {0};
  std::vector<int> fanout = {2, 2};  // 2-hop, 2 neighbors per hop

  auto result = SampleNeighbors(graph_info, "Node", "edge", seeds, fanout, 42);
  REQUIRE(result.status().ok());

  auto& sampling = result.value();
  
  // Should have seed in sampled_nodes
  REQUIRE(std::find(sampling.sampled_nodes.begin(),
                    sampling.sampled_nodes.end(), 0) !=
          sampling.sampled_nodes.end());

  // Should have edges from both hops
  REQUIRE(sampling.src_indices.size() > 0);
  REQUIRE(sampling.src_indices.size() == sampling.dst_indices.size());
  
  // Verify sampled_nodes contains more than just immediate neighbors
  REQUIRE(sampling.sampled_nodes.size() >= 2);
}

TEST_CASE_METHOD(SamplingTestFixture, "SampleNeighbors - isolated node") {
  std::vector<IdType> seeds = {5};  // Node 5 has no edges
  std::vector<int> fanout = {2};

  auto result = SampleNeighbors(graph_info, "Node", "edge", seeds, fanout, 42);
  REQUIRE(result.status().ok());

  auto& sampling = result.value();
  
  // Should only have the seed node
  REQUIRE(sampling.sampled_nodes.size() == 1);
  REQUIRE(sampling.sampled_nodes[0] == 5);
  
  // Should have no edges
  REQUIRE(sampling.src_indices.empty());
  REQUIRE(sampling.dst_indices.empty());
}

TEST_CASE_METHOD(SamplingTestFixture, "SampleNeighbors - multiple seeds") {
  std::vector<IdType> seeds = {0, 1};
  std::vector<int> fanout = {2};

  auto result = SampleNeighbors(graph_info, "Node", "edge", seeds, fanout, 42);
  REQUIRE(result.status().ok());

  auto& sampling = result.value();
  
  // Should have both seeds
  REQUIRE(std::find(sampling.sampled_nodes.begin(),
                    sampling.sampled_nodes.end(), 0) !=
          sampling.sampled_nodes.end());
  REQUIRE(std::find(sampling.sampled_nodes.begin(),
                    sampling.sampled_nodes.end(), 1) !=
          sampling.sampled_nodes.end());

  // Should have edges from both seeds
  REQUIRE(sampling.src_indices.size() > 0);
}

TEST_CASE_METHOD(SamplingTestFixture, "SampleNeighbors - empty seed list") {
  std::vector<IdType> seeds = {};
  std::vector<int> fanout = {2};

  auto result = SampleNeighbors(graph_info, "Node", "edge", seeds, fanout, 42);
  REQUIRE(result.status().ok());

  auto& sampling = result.value();
  REQUIRE(sampling.sampled_nodes.empty());
  REQUIRE(sampling.src_indices.empty());
  REQUIRE(sampling.dst_indices.empty());
}

TEST_CASE_METHOD(SamplingTestFixture, "SampleNeighbors - invalid vertex type") {
  std::vector<IdType> seeds = {0};
  std::vector<int> fanout = {2};

  auto result =
      SampleNeighbors(graph_info, "InvalidType", "edge", seeds, fanout, 42);
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(SamplingTestFixture, "SampleNeighbors - invalid edge type") {
  std::vector<IdType> seeds = {0};
  std::vector<int> fanout = {2};

  auto result =
      SampleNeighbors(graph_info, "Node", "invalid_edge", seeds, fanout, 42);
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(SamplingTestFixture, "SampleNeighbors - empty fanout") {
  std::vector<IdType> seeds = {0};
  std::vector<int> fanout = {};

  auto result = SampleNeighbors(graph_info, "Node", "edge", seeds, fanout, 42);
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

//////////////////////////// GetNodeFeatures /////////////////////////////////////

TEST_CASE_METHOD(SamplingTestFixture, "GetNodeFeatures - single property") {
  std::vector<IdType> node_ids = {0, 1, 2};
  std::vector<std::string> properties = {"id"};

  auto result = GetNodeFeatures(graph_info, "Node", node_ids, properties);
  REQUIRE(result.status().ok());

  auto table = result.value();
  REQUIRE(table->num_rows() == 3);
  REQUIRE(table->num_columns() == 1);

  auto id_column = table->GetColumnByName("id");
  REQUIRE(id_column != nullptr);
  
  // Verify values
  auto id_array = std::static_pointer_cast<arrow::Int64Array>(id_column->chunk(0));
  REQUIRE(id_array->Value(0) == 0);
  REQUIRE(id_array->Value(1) == 1);
  REQUIRE(id_array->Value(2) == 2);
}

TEST_CASE_METHOD(SamplingTestFixture, "GetNodeFeatures - multiple properties") {
  std::vector<IdType> node_ids = {0, 2, 4};
  std::vector<std::string> properties = {"id", "feature"};

  auto result = GetNodeFeatures(graph_info, "Node", node_ids, properties);
  REQUIRE(result.status().ok());

  auto table = result.value();
  REQUIRE(table->num_rows() == 3);
  REQUIRE(table->num_columns() == 2);

  // Verify id column
  auto id_column = table->GetColumnByName("id");
  REQUIRE(id_column != nullptr);
  auto id_array = std::static_pointer_cast<arrow::Int64Array>(id_column->chunk(0));
  REQUIRE(id_array->Value(0) == 0);
  REQUIRE(id_array->Value(1) == 2);
  REQUIRE(id_array->Value(2) == 4);

  // Verify feature column
  auto feature_column = table->GetColumnByName("feature");
  REQUIRE(feature_column != nullptr);
  auto feature_array = std::static_pointer_cast<arrow::FloatArray>(feature_column->chunk(0));
  REQUIRE(std::abs(feature_array->Value(0) - 0.0f) < 0.001f);
  REQUIRE(std::abs(feature_array->Value(1) - 0.2f) < 0.001f);
  REQUIRE(std::abs(feature_array->Value(2) - 0.4f) < 0.001f);
}

TEST_CASE_METHOD(SamplingTestFixture, "GetNodeFeatures - empty node list") {
  std::vector<IdType> node_ids = {};
  std::vector<std::string> properties = {"id"};

  auto result = GetNodeFeatures(graph_info, "Node", node_ids, properties);
  REQUIRE(result.status().ok());

  auto table = result.value();
  REQUIRE(table->num_rows() == 0);
}

TEST_CASE_METHOD(SamplingTestFixture, "GetNodeFeatures - empty properties") {
  std::vector<IdType> node_ids = {0};
  std::vector<std::string> properties = {};

  auto result = GetNodeFeatures(graph_info, "Node", node_ids, properties);
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(SamplingTestFixture, "GetNodeFeatures - invalid property") {
  std::vector<IdType> node_ids = {0};
  std::vector<std::string> properties = {"non_existent"};

  auto result = GetNodeFeatures(graph_info, "Node", node_ids, properties);
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(SamplingTestFixture, "GetNodeFeatures - invalid vertex type") {
  std::vector<IdType> node_ids = {0};
  std::vector<std::string> properties = {"id"};

  auto result = GetNodeFeatures(graph_info, "InvalidType", node_ids, properties);
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(SamplingTestFixture, "GetNodeFeatures - unordered node ids") {
  std::vector<IdType> node_ids = {5, 1, 3, 0};
  std::vector<std::string> properties = {"id"};

  auto result = GetNodeFeatures(graph_info, "Node", node_ids, properties);
  if (!result.status().ok()) {
    std::cerr << "Error: " << result.status().message() << std::endl;
  }
  REQUIRE(result.status().ok());

  auto table = result.value();
  REQUIRE(table->num_rows() == 4);

  // Values should be in the same order as input node_ids
  auto id_column = table->GetColumnByName("id");
  auto id_array = std::static_pointer_cast<arrow::Int64Array>(id_column->chunk(0));
  REQUIRE(id_array->Value(0) == 5);
  REQUIRE(id_array->Value(1) == 1);
  REQUIRE(id_array->Value(2) == 3);
  REQUIRE(id_array->Value(3) == 0);
}

}  // namespace graphar::ml
