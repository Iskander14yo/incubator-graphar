#include <algorithm>
#include <filesystem>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "graphar/filesystem.h"
#include "graphar/graph_info.h"
#include "graphar/ml/chunk_read_manager.h"
#include "graphar/ml/edge_pipeline.h"
#include "graphar/ml/feature_pipeline.h"
#include "graphar/ml/neighbor_sampling.h"
#include "graphar/reader_util.h"
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

class MissingPropertyChunkGraphCopy {
 public:
  explicit MissingPropertyChunkGraphCopy(const std::string& test_data_dir) {
    root_ = std::filesystem::temp_directory_path() /
            "graphar_ml_missing_property_chunk";
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

    const auto chunk_path =
        std::filesystem::path(normalized_prefix + maybe_chunk_path.value());
    REQUIRE(std::filesystem::exists(chunk_path));
    REQUIRE(std::filesystem::remove(chunk_path));
  }

  ~MissingPropertyChunkGraphCopy() {
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

IdType GetVertexNumOrRequire(const std::shared_ptr<GraphInfo>& graph_info,
                             const std::shared_ptr<VertexInfo>& vertex_info) {
  auto vertex_num = util::GetVertexNum(graph_info->GetPrefix(), vertex_info);
  REQUIRE(vertex_num.status().ok());
  return vertex_num.value();
}

IdType GetVertexChunkNumOrRequire(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::shared_ptr<VertexInfo>& vertex_info) {
  auto chunk_num = util::GetVertexChunkNum(graph_info->GetPrefix(), vertex_info);
  REQUIRE(chunk_num.status().ok());
  return chunk_num.value();
}

IdType GetChunkStartNodeId(const std::shared_ptr<GraphInfo>& graph_info,
                           const std::shared_ptr<VertexInfo>& vertex_info,
                           IdType chunk_id) {
  const auto node_id = chunk_id * vertex_info->GetChunkSize();
  REQUIRE(node_id < GetVertexNumOrRequire(graph_info, vertex_info));
  return node_id;
}

Result<std::shared_ptr<arrow::Table>> SubmitAndWaitFeatures(
    const std::shared_ptr<GraphInfo>& graph_info, ChunkReadManager* manager,
    const std::string& vertex_type, const std::vector<IdType>& node_ids,
    const std::vector<std::string>& properties) {
  return GetNodeFeatures(graph_info, vertex_type, node_ids, properties, manager);
}

ChunkReadManager MakeFeatureManagedChunkReader(size_t cursor_count = 1,
                                              size_t trail_capacity = 10) {
  ChunkReadManagerOptions options;
  options.feature_cursor_count = cursor_count;
  options.feature_cursor_trail_capacity_chunks = trail_capacity;
  return ChunkReadManager(options);
}

std::shared_ptr<ChunkReadManager> MakeSharedFeatureManagedChunkReader(
    size_t cursor_count = 1, size_t trail_capacity = 10) {
  ChunkReadManagerOptions options;
  options.feature_cursor_count = cursor_count;
  options.feature_cursor_trail_capacity_chunks = trail_capacity;
  return std::make_shared<ChunkReadManager>(options);
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
                 "SampleNeighbors - sparse fanout when degree exceeds fanout") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);

  auto result =
      SampleNeighbors(graph_info, kVertexType, kEdgeType, {0}, {1}, 777);
  REQUIRE(result.status().ok());

  const auto edges = ToNodeEdges(result.value());
  REQUIRE(edges.size() == 1);
  REQUIRE(edges[0].first == 0);
  REQUIRE(std::set<IdType>{87, 623, 849}.count(edges[0].second) == 1);
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

TEST_CASE_METHOD(GlobalFixture,
                 "SampleNeighbors - edge cursors preserve uncached output") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  ChunkReadManagerOptions options;
  options.edge_cursor_count = 1;
  options.edge_cursor_trail_capacity_chunks = 1;
  ChunkReadManager manager(options);

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

  const auto offset_stats = manager.edge_offset_cursor_stats();
  REQUIRE(offset_stats.cursor_count == 1);
  REQUIRE(offset_stats.requests > 0);
  REQUIRE(offset_stats.requests_completed == offset_stats.requests);
  REQUIRE(offset_stats.requests_failed == 0);

  const auto adj_stats = manager.edge_adj_list_cursor_stats();
  REQUIRE(adj_stats.cursor_count == 1);
  REQUIRE(adj_stats.requests > 0);
  REQUIRE(adj_stats.requests_completed == adj_stats.requests);
  REQUIRE(adj_stats.requests_failed == 0);
}

TEST_CASE_METHOD(
    GlobalFixture,
    "Edge sampling pipeline merges batches that share edge chunk keys") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  auto chunk_manager = std::make_shared<ChunkReadManager>();

  EdgeSamplingPipelineOptions options;
  options.num_readers = 1;
  options.num_processors = 2;
  options.max_active_batches = 4;
  options.max_queued_processor_tasks = 8;
  EdgeSamplingPipelineCoordinator coordinator(chunk_manager, options);

  auto expected = SampleNeighbors(graph_info, kVertexType, kEdgeType, {0, 1},
                                  {3, 2}, 42);
  REQUIRE(expected.status().ok());

  auto first_handle = coordinator.SubmitSeedBatch(
      graph_info, kVertexType, kEdgeType, {0, 1}, {3, 2}, 42);
  auto second_handle = coordinator.SubmitSeedBatch(
      graph_info, kVertexType, kEdgeType, {0, 1}, {3, 2}, 42);
  REQUIRE(first_handle.status().ok());
  REQUIRE(second_handle.status().ok());

  auto first = first_handle.value()->Wait();
  auto second = second_handle.value()->Wait();
  REQUIRE(first.status().ok());
  REQUIRE(second.status().ok());
  REQUIRE(first.value().sampled_nodes == expected.value().sampled_nodes);
  REQUIRE(first.value().src_indices == expected.value().src_indices);
  REQUIRE(first.value().dst_indices == expected.value().dst_indices);
  REQUIRE(first.value().num_sampled_nodes_per_hop ==
          expected.value().num_sampled_nodes_per_hop);
  REQUIRE(first.value().num_sampled_edges_per_hop ==
          expected.value().num_sampled_edges_per_hop);
  REQUIRE(second.value().sampled_nodes == expected.value().sampled_nodes);
  REQUIRE(second.value().src_indices == expected.value().src_indices);
  REQUIRE(second.value().dst_indices == expected.value().dst_indices);

  const auto stats = coordinator.Stats();
  REQUIRE(stats.submitted_batches == 2);
  REQUIRE(stats.completed_batches == 2);
  REQUIRE(stats.offset_chunk_subscriptions == 2);
  REQUIRE(stats.offset_chunk_reads == 1);
  REQUIRE(stats.offset_chunk_reuses == 1);
  REQUIRE(stats.adj_chunk_subscriptions > 0);
  REQUIRE(stats.adj_chunk_reads < stats.adj_chunk_subscriptions);
  REQUIRE(stats.adj_chunk_reuses > 0);
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

TEST_CASE_METHOD(
    GlobalFixture,
    "GetNodeFeatures preserves requested property order across groups") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  auto manager = MakeFeatureManagedChunkReader();
  const std::vector<std::string> properties = {"firstName", "id"};

  auto result = GetNodeFeatures(graph_info, kVertexType, {0, 1}, properties);
  auto managed =
      GetNodeFeatures(graph_info, kVertexType, {0, 1}, properties, &manager);
  REQUIRE(result.status().ok());
  REQUIRE(managed.status().ok());

  REQUIRE(result.value()->ColumnNames() == properties);
  REQUIRE(managed.value()->ColumnNames() == properties);
  REQUIRE(managed.value()->Equals(*result.value()));
}

TEST_CASE_METHOD(GlobalFixture,
                 "GetNodeFeatures - chunk manager matches uncached output") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  auto manager = MakeFeatureManagedChunkReader();

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
  REQUIRE(manager.feature_cursor_stats().requests == 1);
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

/////////////////////////// Feature Cursor //////////////////////////////////////

TEST_CASE_METHOD(
    GlobalFixture,
    "Feature pipeline merges batches that share a feature chunk key") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  auto chunk_manager = MakeSharedFeatureManagedChunkReader(1);

  FeaturePipelineOptions options;
  options.num_readers = 1;
  options.num_stitchers = 2;
  options.max_active_batches = 4;
  options.max_queued_stitch_tasks = 8;
  FeaturePipelineCoordinator coordinator(chunk_manager, options);

  auto expected_first = GetNodeFeatures(graph_info, kVertexType, {0, 1}, {"id"});
  auto expected_second =
      GetNodeFeatures(graph_info, kVertexType, {5, 1, 3, 0}, {"id"});
  REQUIRE(expected_first.status().ok());
  REQUIRE(expected_second.status().ok());

  auto first_handle =
      coordinator.SubmitSampledBatch(graph_info, kVertexType, {0, 1}, {"id"});
  auto second_handle = coordinator.SubmitSampledBatch(
      graph_info, kVertexType, {5, 1, 3, 0}, {"id"});
  REQUIRE(first_handle.status().ok());
  REQUIRE(second_handle.status().ok());

  auto first = first_handle.value()->Wait();
  auto second = second_handle.value()->Wait();
  REQUIRE(first.status().ok());
  REQUIRE(second.status().ok());
  REQUIRE(first.value()->Equals(*expected_first.value()));
  REQUIRE(second.value()->Equals(*expected_second.value()));

  const auto pipeline_stats = coordinator.Stats();
  REQUIRE(pipeline_stats.submitted_batches == 2);
  REQUIRE(pipeline_stats.completed_batches == 2);
  REQUIRE(pipeline_stats.active_batches_current == 0);
  REQUIRE(pipeline_stats.pending_batches_peak >= 1);
  REQUIRE(pipeline_stats.active_chunk_keys_current == 0);
  REQUIRE(pipeline_stats.active_chunk_keys_peak == 1);
  REQUIRE(pipeline_stats.read_queue_current == 0);
  REQUIRE(pipeline_stats.stitch_queue_current == 0);
  REQUIRE(pipeline_stats.chunk_subscriptions == 2);
  REQUIRE(pipeline_stats.chunk_reads == 1);
  REQUIRE(pipeline_stats.chunk_reuses == 1);
  REQUIRE(pipeline_stats.stitch_tasks == 2);

  const auto cursor_stats = chunk_manager->feature_cursor_stats();
  REQUIRE(cursor_stats.requests == 2);
  REQUIRE(cursor_stats.requests_completed == 2);
  REQUIRE(cursor_stats.requests_failed == 0);
  REQUIRE(cursor_stats.chunks_read == 1);
  REQUIRE(cursor_stats.chunks_served == 1);
  REQUIRE(cursor_stats.rows_served == 6);
  REQUIRE(cursor_stats.batches_served == 2);
}

TEST_CASE_METHOD(
    GlobalFixture,
    "Cursor-backed feature manager matches direct GetNodeFeatures on ldbc_sample") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  auto chunk_manager = MakeFeatureManagedChunkReader(2);

  const std::vector<IdType> node_ids = {5, 1, 3, 0};
  const std::vector<std::string> properties = {"id", "firstName"};

  auto expected = GetNodeFeatures(graph_info, kVertexType, node_ids, properties);
  auto managed = SubmitAndWaitFeatures(graph_info, &chunk_manager, kVertexType,
                                       node_ids, properties);
  REQUIRE(expected.status().ok());
  REQUIRE(managed.status().ok());
  REQUIRE(managed.value()->Equals(*expected.value()));

  const auto stats = chunk_manager.feature_cursor_stats();
  REQUIRE(stats.cursor_count == 2);
  REQUIRE(stats.requests == 1);
  REQUIRE(stats.requests_completed == 1);
  REQUIRE(stats.requests_failed == 0);
  REQUIRE(stats.chunks_read > 0);
  REQUIRE(stats.batches_served > 0);
}

TEST_CASE_METHOD(
    GlobalFixture,
    "Cursor-backed feature manager handles concurrent requests for the same chunk") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  auto vertex_info = graph_info->GetVertexInfo(kVertexType);
  REQUIRE(vertex_info != nullptr);
  REQUIRE(vertex_info->GetChunkSize() >= 6);

  auto chunk_manager = MakeFeatureManagedChunkReader(1);
  const std::vector<std::vector<IdType>> node_requests = {
      {0, 1}, {5, 1, 3, 0}, {2}, {4, 2, 1}, {0, 5}, {3, 4}};
  std::vector<std::shared_ptr<arrow::Table>> expected_tables(
      node_requests.size());
  for (size_t i = 0; i < node_requests.size(); ++i) {
    auto expected =
        GetNodeFeatures(graph_info, kVertexType, node_requests[i], {"id"});
    REQUIRE(expected.status().ok());
    expected_tables[i] = expected.value();
  }

  std::promise<void> start_promise;
  auto start = start_promise.get_future().share();
  std::vector<std::shared_ptr<arrow::Table>> actual_tables(
      node_requests.size());
  std::vector<Status> statuses(
      node_requests.size(), Status::UnknownError("request not started"));
  std::vector<std::thread> threads;
  threads.reserve(node_requests.size());

  for (size_t i = 0; i < node_requests.size(); ++i) {
    threads.emplace_back([&, i]() {
      start.wait();
      auto result = SubmitAndWaitFeatures(graph_info, &chunk_manager,
                                          kVertexType, node_requests[i],
                                          {"id"});
      statuses[i] = result.status();
      if (!result.has_error()) {
        actual_tables[i] = result.value();
      }
    });
  }

  start_promise.set_value();
  for (auto& thread : threads) {
    thread.join();
  }

  uint64_t total_rows_served = 0;
  for (size_t i = 0; i < node_requests.size(); ++i) {
    REQUIRE(statuses[i].ok());
    REQUIRE(actual_tables[i] != nullptr);
    REQUIRE(actual_tables[i]->Equals(*expected_tables[i]));
    total_rows_served += static_cast<uint64_t>(node_requests[i].size());
  }

  const auto stats = chunk_manager.feature_cursor_stats();
  REQUIRE(stats.requests == node_requests.size());
  REQUIRE(stats.requests_completed == node_requests.size());
  REQUIRE(stats.requests_failed == 0);
  REQUIRE(stats.chunks_read == 1);
  REQUIRE(stats.chunks_served == 1);
  REQUIRE(stats.chunk_order_wraps == 0);
  REQUIRE(stats.rows_served == total_rows_served);
  REQUIRE(stats.batches_served == node_requests.size());
}

TEST_CASE_METHOD(
    GlobalFixture,
    "Cursor-backed feature manager serves disjoint chunks across cursors") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  auto vertex_info = graph_info->GetVertexInfo(kVertexType);
  REQUIRE(vertex_info != nullptr);

  const auto chunk_count = GetVertexChunkNumOrRequire(graph_info, vertex_info);
  REQUIRE(chunk_count >= 3);
  const std::vector<IdType> chunk_ids = {0, chunk_count / 2, chunk_count - 1};
  REQUIRE(chunk_ids[0] < chunk_ids[1]);
  REQUIRE(chunk_ids[1] < chunk_ids[2]);

  auto chunk_manager = MakeFeatureManagedChunkReader(3);
  std::promise<void> start_promise;
  auto start = start_promise.get_future().share();
  std::vector<std::shared_ptr<arrow::Table>> results(chunk_ids.size());
  std::vector<Status> statuses(
      chunk_ids.size(), Status::UnknownError("request not started"));
  std::vector<std::thread> threads;
  threads.reserve(chunk_ids.size());

  for (size_t i = 0; i < chunk_ids.size(); ++i) {
    threads.emplace_back([&, i]() {
      start.wait();
      const auto node_id =
          GetChunkStartNodeId(graph_info, vertex_info, chunk_ids[i]);
      auto result = SubmitAndWaitFeatures(graph_info, &chunk_manager,
                                          kVertexType, {node_id}, {"id"});
      statuses[i] = result.status();
      if (!result.has_error()) {
        results[i] = result.value();
      }
    });
  }

  start_promise.set_value();
  for (auto& thread : threads) {
    thread.join();
  }

  for (size_t i = 0; i < chunk_ids.size(); ++i) {
    REQUIRE(statuses[i].ok());
    auto expected =
        GetNodeFeatures(graph_info, kVertexType,
                        {GetChunkStartNodeId(graph_info, vertex_info,
                                             chunk_ids[i])},
                        {"id"});
    REQUIRE(expected.status().ok());
    REQUIRE(results[i] != nullptr);
    REQUIRE(results[i]->Equals(*expected.value()));
  }

  const auto stats = chunk_manager.feature_cursor_stats();
  REQUIRE(stats.cursor_count == 3);
  REQUIRE(stats.requests == chunk_ids.size());
  REQUIRE(stats.requests_completed == chunk_ids.size());
  REQUIRE(stats.requests_failed == 0);
  REQUIRE(stats.chunks_read == chunk_ids.size());
  REQUIRE(stats.chunks_served == chunk_ids.size());
  REQUIRE(stats.rows_served == chunk_ids.size());
  REQUIRE(stats.batches_served == chunk_ids.size());
  REQUIRE(stats.trail_hits == 0);
  REQUIRE(stats.trail_misses == chunk_ids.size());
}

TEST_CASE_METHOD(
    GlobalFixture,
    "Cursor-backed feature manager wraps chunk order for lower chunk ids") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  auto vertex_info = graph_info->GetVertexInfo(kVertexType);
  REQUIRE(vertex_info != nullptr);

  const auto chunk_count = GetVertexChunkNumOrRequire(graph_info, vertex_info);
  REQUIRE(chunk_count >= 2);

  auto chunk_manager = MakeFeatureManagedChunkReader(1);
  const IdType high_chunk_id = chunk_count - 1;
  const IdType low_chunk_id = 0;
  const IdType high_node_id =
      GetChunkStartNodeId(graph_info, vertex_info, high_chunk_id);
  const IdType low_node_id =
      GetChunkStartNodeId(graph_info, vertex_info, low_chunk_id);

  auto first_expected =
      GetNodeFeatures(graph_info, kVertexType, {high_node_id}, {"id"});
  auto second_expected =
      GetNodeFeatures(graph_info, kVertexType, {low_node_id}, {"id"});

  auto first =
      SubmitAndWaitFeatures(graph_info, &chunk_manager, kVertexType,
                            {high_node_id}, {"id"});
  auto second =
      SubmitAndWaitFeatures(graph_info, &chunk_manager, kVertexType,
                            {low_node_id}, {"id"});
  REQUIRE(first_expected.status().ok());
  REQUIRE(second_expected.status().ok());
  REQUIRE(first.status().ok());
  REQUIRE(second.status().ok());
  REQUIRE(first.value()->Equals(*first_expected.value()));
  REQUIRE(second.value()->Equals(*second_expected.value()));

  const auto stats = chunk_manager.feature_cursor_stats();
  REQUIRE(stats.requests == 2);
  REQUIRE(stats.requests_completed == 2);
  REQUIRE(stats.requests_failed == 0);
  REQUIRE(stats.chunks_read == 2);
  REQUIRE(stats.chunks_served == 2);
  REQUIRE(stats.chunk_order_wraps == 1);
}

TEST_CASE_METHOD(
    GlobalFixture,
    "Cursor-backed feature manager reuses trail cache for repeated chunk request") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  auto vertex_info = graph_info->GetVertexInfo(kVertexType);
  REQUIRE(vertex_info != nullptr);
  REQUIRE(vertex_info->GetChunkSize() >= 4);

  auto chunk_manager = MakeFeatureManagedChunkReader(1, 1);

  auto first_expected = GetNodeFeatures(graph_info, kVertexType, {0, 1}, {"id"});
  auto second_expected =
      GetNodeFeatures(graph_info, kVertexType, {2, 3}, {"id"});
  auto first =
      SubmitAndWaitFeatures(graph_info, &chunk_manager, kVertexType, {0, 1},
                            {"id"});
  auto second =
      SubmitAndWaitFeatures(graph_info, &chunk_manager, kVertexType, {2, 3},
                            {"id"});
  REQUIRE(first_expected.status().ok());
  REQUIRE(second_expected.status().ok());
  REQUIRE(first.status().ok());
  REQUIRE(second.status().ok());

  REQUIRE(first.value()->Equals(*first_expected.value()));
  REQUIRE(second.value()->Equals(*second_expected.value()));

  const auto stats = chunk_manager.feature_cursor_stats();
  REQUIRE(stats.requests == 2);
  REQUIRE(stats.requests_completed == 2);
  REQUIRE(stats.requests_failed == 0);
  REQUIRE(stats.chunks_read == 1);
  REQUIRE(stats.chunks_served == 1);
  REQUIRE(stats.chunk_order_wraps == 0);
  REQUIRE(stats.rows_served == 4);
  REQUIRE(stats.batches_served == 2);
  REQUIRE(stats.trail_hits == 1);
  REQUIRE(stats.trail_misses == 1);
}

TEST_CASE_METHOD(GlobalFixture,
                 "Cursor-backed feature manager leaves empty node list fast path unchanged") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  auto chunk_manager = MakeFeatureManagedChunkReader(3);

  auto result = SubmitAndWaitFeatures(graph_info, &chunk_manager, kVertexType,
                                      {}, {"id"});
  REQUIRE(result.status().ok());
  REQUIRE(result.value()->num_rows() == 0);

  const auto stats = chunk_manager.feature_cursor_stats();
  REQUIRE(stats.cursor_count == 3);
  REQUIRE(stats.requests == 0);
  REQUIRE(stats.requests_completed == 0);
  REQUIRE(stats.requests_failed == 0);
  REQUIRE(stats.chunks_read == 0);
  REQUIRE(stats.chunks_served == 0);
  REQUIRE(stats.rows_served == 0);
  REQUIRE(stats.batches_served == 0);
}

TEST_CASE_METHOD(GlobalFixture,
                 "Cursor-backed feature manager rejects requests after shutdown") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  auto chunk_manager = MakeFeatureManagedChunkReader(1);

  chunk_manager.Shutdown();

  auto result =
      SubmitAndWaitFeatures(graph_info, &chunk_manager, kVertexType, {0},
                            {"id"});
  REQUIRE(result.has_error());
  REQUIRE(result.status().IsInvalid());
  REQUIRE(result.status().message().find("shut down") != std::string::npos);

  const auto stats = chunk_manager.feature_cursor_stats();
  REQUIRE(stats.requests == 1);
  REQUIRE(stats.requests_completed == 0);
  REQUIRE(stats.requests_failed == 1);
}

TEST_CASE_METHOD(GlobalFixture,
                 "Cursor-backed feature manager propagates chunk read failures") {
  MissingPropertyChunkGraphCopy graph_copy(test_data_dir);
  auto chunk_manager = MakeFeatureManagedChunkReader(1);

  auto result = SubmitAndWaitFeatures(graph_copy.graph_info(), &chunk_manager,
                                      kVertexType, {0}, {"id"});
  REQUIRE(result.has_error());

  const auto stats = chunk_manager.feature_cursor_stats();
  REQUIRE(stats.requests == 1);
  REQUIRE(stats.requests_completed == 0);
  REQUIRE(stats.requests_failed == 1);
  REQUIRE(stats.chunks_read == 1);
  REQUIRE(stats.chunks_served == 0);
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

  auto chunk_manager = MakeFeatureManagedChunkReader(1);
  auto managed = SubmitAndWaitFeatures(graph_copy.graph_info(), &chunk_manager,
                                       kVertexType, node_ids, properties);
  REQUIRE(managed.status().ok());
  REQUIRE(managed.value()->Equals(*expected.value()));
}

}  // namespace graphar::ml
