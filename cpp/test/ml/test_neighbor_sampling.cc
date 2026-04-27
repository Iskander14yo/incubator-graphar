#include <algorithm>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "graphar/filesystem.h"
#include "graphar/graph_info.h"
#include "graphar/ml/chunk_read_manager.h"
#include "graphar/ml/feature_cursor.h"
#include "graphar/ml/neighbor_sampling.h"
#include "graphar/writer_util.h"

#include <catch2/catch_test_macros.hpp>

#include "../util.h"

namespace graphar::ml {
namespace {

constexpr const char* kGraphPath = "/ldbc_sample/parquet/ldbc_sample.graph.yml";
constexpr const char* kVertexType = "person";
constexpr const char* kEdgeType = "knows";

std::shared_ptr<GraphInfo> LoadLdbcSampleGraph(const std::string& test_data_dir) {
  auto maybe_graph_info = GraphInfo::Load(test_data_dir + kGraphPath);
  REQUIRE(maybe_graph_info.status().ok());
  return maybe_graph_info.value();
}

void CopyDirectoryTree(const std::filesystem::path& source,
                       const std::filesystem::path& destination) {
  std::filesystem::create_directories(destination);
  for (const auto& entry : std::filesystem::recursive_directory_iterator(
           source)) {
    const auto relative = std::filesystem::relative(entry.path(), source);
    const auto target = destination / relative;
    if (entry.is_directory()) {
      std::filesystem::create_directories(target);
    } else if (entry.is_regular_file()) {
      std::filesystem::create_directories(target.parent_path());
      std::filesystem::copy_file(
          entry.path(), target,
          std::filesystem::copy_options::overwrite_existing);
    }
  }
}

class MultiRowGroupGraphCopy {
 public:
  explicit MultiRowGroupGraphCopy(const std::string& test_data_dir) {
    root_ = std::filesystem::temp_directory_path() /
            "graphar_ml_multi_row_group";
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);

    const auto source_root =
        std::filesystem::path(test_data_dir) / "ldbc_sample" / "parquet";
    CopyDirectoryTree(source_root, root_);

    auto maybe_graph_info =
        GraphInfo::Load((root_ / "ldbc_sample.graph.yml").string());
    REQUIRE(maybe_graph_info.status().ok());
    graph_info_ = maybe_graph_info.value();

    auto vertex_info = graph_info_->GetVertexInfo(kVertexType);
    REQUIRE(vertex_info != nullptr);
    auto property_group = vertex_info->GetPropertyGroup("id");
    REQUIRE(property_group != nullptr);

    auto maybe_chunk_path = vertex_info->GetFilePath(property_group, 0);
    REQUIRE(maybe_chunk_path.status().ok());

    std::string normalized_prefix;
    auto maybe_fs = FileSystemFromUriOrPath(graph_info_->GetPrefix(),
                                            &normalized_prefix);
    REQUIRE(maybe_fs.status().ok());
    auto fs = maybe_fs.value();
    const auto chunk_path = normalized_prefix + maybe_chunk_path.value();

    auto maybe_table =
        fs->ReadFileToTable(chunk_path, property_group->GetFileType());
    REQUIRE(maybe_table.status().ok());

    auto writer_options =
        WriterOptions::ParquetOptionBuilder().max_row_group_length(10).build();
    REQUIRE(fs->WriteTableToFile(maybe_table.value(), FileType::PARQUET,
                                 chunk_path, writer_options)
                .ok());

    auto maybe_rewritten =
        fs->ReadFileToTable(chunk_path, property_group->GetFileType());
    REQUIRE(maybe_rewritten.status().ok());
    auto column = maybe_rewritten.value()->GetColumnByName("id");
    REQUIRE(column != nullptr);
    REQUIRE(column->num_chunks() > 1);
  }

  ~MultiRowGroupGraphCopy() {
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
  }

  const std::shared_ptr<GraphInfo>& graph_info() const { return graph_info_; }

 private:
  std::filesystem::path root_;
  std::shared_ptr<GraphInfo> graph_info_;
};

std::vector<std::pair<IdType, IdType>> ToNodeEdges(
    const SamplingResult& sampling) {
  std::vector<std::pair<IdType, IdType>> edges;
  edges.reserve(sampling.src_indices.size());
  for (size_t i = 0; i < sampling.src_indices.size(); ++i) {
    edges.emplace_back(sampling.sampled_nodes[sampling.src_indices[i]],
                       sampling.sampled_nodes[sampling.dst_indices[i]]);
  }
  return edges;
}

std::vector<IdType> Int64Values(
    const std::shared_ptr<arrow::ChunkedArray>& column) {
  std::vector<IdType> values;
  values.reserve(column->length());
  for (const auto& chunk : column->chunks()) {
    auto array = std::static_pointer_cast<arrow::Int64Array>(chunk);
    for (int64_t i = 0; i < array->length(); ++i) {
      values.push_back(array->Value(i));
    }
  }
  return values;
}

std::vector<std::string> StringValues(
    const std::shared_ptr<arrow::ChunkedArray>& column) {
  std::vector<std::string> values;
  values.reserve(column->length());
  for (const auto& chunk : column->chunks()) {
    if (chunk->type()->id() == arrow::Type::STRING) {
      auto array = std::static_pointer_cast<arrow::StringArray>(chunk);
      for (int64_t i = 0; i < array->length(); ++i) {
        values.push_back(array->GetString(i));
      }
      continue;
    }
    auto array = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
    for (int64_t i = 0; i < array->length(); ++i) {
      values.push_back(array->GetString(i));
    }
  }
  return values;
}

}  // namespace

//////////////////////////// SampleNeighbors /////////////////////////////////////

TEST_CASE_METHOD(GlobalFixture, "SampleNeighbors - single hop on ldbc_sample") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = SampleNeighbors(graph_info, kVertexType, kEdgeType, {0}, {10}, 42);
  REQUIRE(result.status().ok());

  const auto& sampling = result.value();
  REQUIRE(sampling.sampled_nodes == std::vector<IdType>{0, 87, 623, 849});
  REQUIRE(sampling.num_sampled_nodes_per_hop == std::vector<IdType>{1, 3});
  REQUIRE(sampling.num_sampled_edges_per_hop == std::vector<IdType>{3});
  REQUIRE(ToNodeEdges(sampling) == std::vector<std::pair<IdType, IdType>>{
                                      {0, 87}, {0, 623}, {0, 849}});
}

TEST_CASE_METHOD(GlobalFixture, "SampleNeighbors - fanout respected on ldbc_sample") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = SampleNeighbors(graph_info, kVertexType, kEdgeType, {0}, {2}, 42);
  REQUIRE(result.status().ok());

  const auto& sampling = result.value();
  const auto edges = ToNodeEdges(sampling);
  const std::set<IdType> expected_neighbors = {87, 623, 849};

  REQUIRE(edges.size() == 2);
  REQUIRE(sampling.src_indices.size() == sampling.dst_indices.size());
  REQUIRE(sampling.sampled_nodes.size() == 3);
  for (const auto& [src, dst] : edges) {
    REQUIRE(src == 0);
    REQUIRE(expected_neighbors.count(dst) == 1);
  }
}

TEST_CASE_METHOD(GlobalFixture,
                 "SampleNeighbors - deterministic with seed on ldbc_sample") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result1 =
      SampleNeighbors(graph_info, kVertexType, kEdgeType, {0}, {2}, 12345);
  auto result2 =
      SampleNeighbors(graph_info, kVertexType, kEdgeType, {0}, {2}, 12345);
  REQUIRE(result1.status().ok());
  REQUIRE(result2.status().ok());

  REQUIRE(result1.value().sampled_nodes == result2.value().sampled_nodes);
  REQUIRE(result1.value().src_indices == result2.value().src_indices);
  REQUIRE(result1.value().dst_indices == result2.value().dst_indices);
}

TEST_CASE_METHOD(GlobalFixture, "SampleNeighbors - multi-hop on ldbc_sample") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result =
      SampleNeighbors(graph_info, kVertexType, kEdgeType, {0}, {3, 2}, 42);
  REQUIRE(result.status().ok());

  const auto& sampling = result.value();
  const auto edges = ToNodeEdges(sampling);

  REQUIRE(sampling.src_indices.size() == 9);
  REQUIRE(sampling.dst_indices.size() == 9);
  REQUIRE(std::count_if(edges.begin(), edges.end(),
                        [](const auto& edge) { return edge.first == 0; }) == 3);
  REQUIRE(std::any_of(edges.begin(), edges.end(),
                      [](const auto& edge) { return edge.first != 0; }));
  REQUIRE(std::find(sampling.sampled_nodes.begin(), sampling.sampled_nodes.end(),
                    87) != sampling.sampled_nodes.end());
  REQUIRE(std::find(sampling.sampled_nodes.begin(), sampling.sampled_nodes.end(),
                    623) != sampling.sampled_nodes.end());
  REQUIRE(std::find(sampling.sampled_nodes.begin(), sampling.sampled_nodes.end(),
                    849) != sampling.sampled_nodes.end());
}

TEST_CASE_METHOD(GlobalFixture, "SampleNeighbors - isolated node on ldbc_sample") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = SampleNeighbors(graph_info, kVertexType, kEdgeType, {2}, {5}, 42);
  REQUIRE(result.status().ok());

  const auto& sampling = result.value();
  REQUIRE(sampling.sampled_nodes == std::vector<IdType>{2});
  REQUIRE(sampling.src_indices.empty());
  REQUIRE(sampling.dst_indices.empty());
}

TEST_CASE_METHOD(GlobalFixture, "SampleNeighbors - multiple seeds on ldbc_sample") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result =
      SampleNeighbors(graph_info, kVertexType, kEdgeType, {0, 1}, {10}, 42);
  REQUIRE(result.status().ok());

  const auto& sampling = result.value();
  REQUIRE(sampling.sampled_nodes ==
          std::vector<IdType>{0, 1, 87, 623, 849, 58, 318, 538, 539, 696});
  REQUIRE(sampling.num_sampled_nodes_per_hop == std::vector<IdType>{2, 8});
  REQUIRE(sampling.num_sampled_edges_per_hop == std::vector<IdType>{8});
  REQUIRE(ToNodeEdges(sampling) == std::vector<std::pair<IdType, IdType>>{
                                      {0, 87},  {0, 623}, {0, 849}, {1, 58},
                                      {1, 318}, {1, 538}, {1, 539}, {1, 696}});
}

TEST_CASE_METHOD(GlobalFixture, "SampleNeighbors - empty seed list") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = SampleNeighbors(graph_info, kVertexType, kEdgeType, {}, {5}, 42);
  REQUIRE(result.status().ok());

  const auto& sampling = result.value();
  REQUIRE(sampling.sampled_nodes.empty());
  REQUIRE(sampling.src_indices.empty());
  REQUIRE(sampling.dst_indices.empty());
  REQUIRE(sampling.num_sampled_nodes_per_hop == std::vector<IdType>{0, 0});
  REQUIRE(sampling.num_sampled_edges_per_hop == std::vector<IdType>{0});
}

TEST_CASE_METHOD(GlobalFixture, "SampleNeighbors - invalid vertex type") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result =
      SampleNeighbors(graph_info, "invalid_person", kEdgeType, {0}, {5}, 42);
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(GlobalFixture, "SampleNeighbors - invalid edge type") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result =
      SampleNeighbors(graph_info, kVertexType, "invalid_edge", {0}, {5}, 42);
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(GlobalFixture, "SampleNeighbors - empty fanout") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = SampleNeighbors(graph_info, kVertexType, kEdgeType, {0}, {}, 42);
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(GlobalFixture, "SampleNeighbors - negative fanout") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = SampleNeighbors(graph_info, kVertexType, kEdgeType, {0}, {-1}, 42);
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(GlobalFixture,
                 "SampleNeighbors - chunk manager matches uncached output") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  ChunkReadManager manager;

  auto uncached =
      SampleNeighbors(graph_info, kVertexType, kEdgeType, {0, 1}, {3, 2}, 42);
  auto managed = SampleNeighbors(graph_info, kVertexType, kEdgeType, {0, 1},
                                 {3, 2}, 42, &manager);
  REQUIRE(uncached.status().ok());
  REQUIRE(managed.status().ok());

  REQUIRE(managed.value().sampled_nodes == uncached.value().sampled_nodes);
  REQUIRE(managed.value().src_indices == uncached.value().src_indices);
  REQUIRE(managed.value().dst_indices == uncached.value().dst_indices);
  REQUIRE(managed.value().num_sampled_nodes_per_hop ==
          uncached.value().num_sampled_nodes_per_hop);
  REQUIRE(managed.value().num_sampled_edges_per_hop ==
          uncached.value().num_sampled_edges_per_hop);

  const auto stats = manager.stats();
  REQUIRE(stats.requests > 0);
  REQUIRE(stats.leaders == stats.requests);
  REQUIRE(stats.completed == stats.requests);
  REQUIRE(stats.failed == 0);
}

//////////////////////////// GetNodeFeatures /////////////////////////////////////

TEST_CASE_METHOD(GlobalFixture, "GetNodeFeatures - single property on ldbc_sample") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = GetNodeFeatures(graph_info, kVertexType, {0, 1, 2}, {"id"});
  REQUIRE(result.status().ok());

  auto table = result.value();
  REQUIRE(table->num_rows() == 3);
  REQUIRE(table->num_columns() == 1);

  auto id_column = table->GetColumnByName("id");
  REQUIRE(id_column != nullptr);
  REQUIRE(Int64Values(id_column) ==
          std::vector<IdType>{933, 6597069767117, 10995116278700});
}

TEST_CASE_METHOD(GlobalFixture,
                 "GetNodeFeatures - multiple properties on ldbc_sample") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result =
      GetNodeFeatures(graph_info, kVertexType, {0, 1}, {"id", "firstName"});
  REQUIRE(result.status().ok());

  auto table = result.value();
  REQUIRE(table->num_rows() == 2);
  REQUIRE(table->num_columns() == 2);

  auto id_column = table->GetColumnByName("id");
  REQUIRE(id_column != nullptr);
  REQUIRE(Int64Values(id_column) == std::vector<IdType>{933, 6597069767117});

  auto first_name_column = table->GetColumnByName("firstName");
  REQUIRE(first_name_column != nullptr);
  REQUIRE(StringValues(first_name_column) ==
          std::vector<std::string>{"Mahinda", "Eli"});
}

TEST_CASE_METHOD(GlobalFixture,
                 "GetNodeFeatures - chunk manager matches uncached output") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  ChunkReadManager manager;

  auto uncached =
      GetNodeFeatures(graph_info, kVertexType, {5, 1, 3, 0}, {"id", "firstName"});
  auto managed = GetNodeFeatures(graph_info, kVertexType, {5, 1, 3, 0},
                                 {"id", "firstName"}, &manager);
  REQUIRE(uncached.status().ok());
  REQUIRE(managed.status().ok());

  REQUIRE(managed.value()->Equals(*uncached.value()));
  const auto stats = manager.stats();
  REQUIRE(stats.requests > 0);
  REQUIRE(stats.leaders == stats.requests);
  REQUIRE(stats.completed == stats.requests);
  REQUIRE(stats.failed == 0);
}

TEST_CASE_METHOD(GlobalFixture, "GetNodeFeatures - empty node list") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = GetNodeFeatures(graph_info, kVertexType, {}, {"id"});
  REQUIRE(result.status().ok());

  auto table = result.value();
  REQUIRE(table->num_rows() == 0);
}

TEST_CASE_METHOD(GlobalFixture, "GetNodeFeatures - empty properties") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = GetNodeFeatures(graph_info, kVertexType, {0}, {});
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(GlobalFixture, "GetNodeFeatures - invalid property") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = GetNodeFeatures(graph_info, kVertexType, {0},
                                {"nonexistent_property"});
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(GlobalFixture, "GetNodeFeatures - invalid vertex type") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = GetNodeFeatures(graph_info, "invalid_person", {0}, {"id"});
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
}

TEST_CASE_METHOD(GlobalFixture, "GetNodeFeatures - unordered node ids") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result = GetNodeFeatures(graph_info, kVertexType, {5, 1, 3, 0}, {"id"});
  REQUIRE(result.status().ok());

  auto table = result.value();
  auto id_column = table->GetColumnByName("id");
  REQUIRE(id_column != nullptr);
  REQUIRE(Int64Values(id_column) == std::vector<IdType>{
                                       28587302322727, 6597069767117,
                                       21990232556027, 933});
}

TEST_CASE_METHOD(GlobalFixture,
                 "Feature reads handle multi-row-group parquet chunks") {
  auto baseline_graph_info = LoadLdbcSampleGraph(test_data_dir);
  MultiRowGroupGraphCopy graph_copy(test_data_dir);
  const std::vector<IdType> node_ids = {0, 15, 27, 99};
  const std::vector<std::string> properties = {"id"};

  auto expected =
      GetNodeFeatures(baseline_graph_info, kVertexType, node_ids, properties);
  REQUIRE(expected.status().ok());

  auto direct = GetNodeFeatures(graph_copy.graph_info(), kVertexType, node_ids,
                                properties);
  REQUIRE(direct.status().ok());
  REQUIRE(direct.value()->Equals(*expected.value()));

  auto chunk_manager = std::make_shared<ChunkReadManager>();
  FeatureScanCoordinator coordinator(graph_copy.graph_info(), chunk_manager);
  auto handle = coordinator.Submit(kVertexType, node_ids, properties);
  REQUIRE(handle.status().ok());

  auto coordinated = handle.value()->Wait();
  REQUIRE(coordinated.status().ok());
  REQUIRE(coordinated.value()->Equals(*expected.value()));
}

}  // namespace graphar::ml
