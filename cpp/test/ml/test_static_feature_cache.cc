#include <memory>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "graphar/graph_info.h"
#include "graphar/ml/neighbor_sampling.h"
#include "graphar/ml/static_feature_cache.h"

#include <catch2/catch_test_macros.hpp>

#include "../util.h"

namespace graphar::ml {
namespace {

std::vector<int64_t> Int64Column(const std::shared_ptr<arrow::Table>& table,
                                 const char* name) {
  auto col = table->GetColumnByName(name);
  REQUIRE(col != nullptr);
  std::vector<int64_t> out;
  for (const auto& chunk : col->chunks()) {
    auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
    for (int64_t i = 0; i < arr->length(); ++i) {
      out.push_back(arr->Value(i));
    }
  }
  return out;
}

}  // namespace

TEST_CASE_METHOD(GlobalFixture, "StaticFeatureCache - Lookup hit matches uncached") {
  auto maybe_graph = GraphInfo::Load(test_data_dir + "/ldbc_sample/parquet/ldbc_sample.graph.yml");
  REQUIRE(maybe_graph.status().ok());
  auto graph = maybe_graph.value();

  StaticFeatureCache cache(graph);
  std::vector<IdType> pin_ids = {0, 1, 2, 5};
  REQUIRE(cache.Pin("person", pin_ids, {"id"}).ok());

  std::vector<IdType> query = {5, 1, 0, 2};
  auto plain = GetNodeFeatures(graph, "person", query, {"id"});
  REQUIRE(plain.status().ok());

  auto lr = cache.Lookup("person", query, {"id"});
  REQUIRE(lr.status().ok());
  const auto& res = lr.value();
  REQUIRE(res.miss_positions.empty());
  REQUIRE(res.hit_positions.size() == query.size());
  REQUIRE(static_cast<size_t>(res.hits->num_rows()) == query.size());
  REQUIRE(Int64Column(res.hits, "id") == Int64Column(plain.value(), "id"));
}

TEST_CASE_METHOD(GlobalFixture, "StaticFeatureCache - mixed hit and miss") {
  auto maybe_graph = GraphInfo::Load(test_data_dir + "/ldbc_sample/parquet/ldbc_sample.graph.yml");
  REQUIRE(maybe_graph.status().ok());
  auto graph = maybe_graph.value();

  StaticFeatureCache cache(graph);
  REQUIRE(cache.Pin("person", {0, 1, 2}, {"id"}).ok());

  std::vector<IdType> query = {0, 999999999, 1};
  auto lr = cache.Lookup("person", query, {"id"});
  REQUIRE(lr.status().ok());
  const auto& res = lr.value();
  REQUIRE(res.miss_positions.size() == 1);
  REQUIRE(res.miss_positions[0] == 1);
  REQUIRE(res.miss_node_ids[0] == 999999999);
  REQUIRE(res.hit_positions.size() == 2);
  REQUIRE(static_cast<size_t>(res.hits->num_rows()) == 2);
  auto plain = GetNodeFeatures(graph, "person", std::vector<IdType>({0, 1}), {"id"});
  REQUIRE(plain.status().ok());
  REQUIRE(Int64Column(res.hits, "id") == Int64Column(plain.value(), "id"));
  REQUIRE(cache.hits() == 2);
  REQUIRE(cache.misses() == 1);
}

TEST_CASE_METHOD(GlobalFixture, "StaticFeatureCache - Pin idempotent") {
  auto maybe_graph = GraphInfo::Load(test_data_dir + "/ldbc_sample/parquet/ldbc_sample.graph.yml");
  REQUIRE(maybe_graph.status().ok());
  auto graph = maybe_graph.value();

  StaticFeatureCache cache(graph);
  std::vector<IdType> ids = {0, 1, 2, 3};
  REQUIRE(cache.Pin("person", ids, {"id"}).ok());
  size_t n1 = cache.num_nodes();
  size_t s1 = cache.size_bytes();
  REQUIRE(n1 > 0);
  REQUIRE(s1 > 0);

  REQUIRE(cache.Pin("person", ids, {"id"}).ok());
  REQUIRE(cache.num_nodes() == n1);
  REQUIRE(cache.size_bytes() == s1);
}

TEST_CASE_METHOD(GlobalFixture,
                 "GetNodeFeatures static_cache matches uncached (interleaved)") {
  auto maybe_graph = GraphInfo::Load(test_data_dir + "/ldbc_sample/parquet/ldbc_sample.graph.yml");
  REQUIRE(maybe_graph.status().ok());
  auto graph = maybe_graph.value();

  StaticFeatureCache cache(graph);
  REQUIRE(cache.Pin("person", std::vector<IdType>({0, 2, 4, 10}), {"id"}).ok());

  // 7 is valid but unpinned → miss; all IDs stay within the graph (e.g. ldbc ~903 persons).
  std::vector<IdType> q = {10, 5, 0, 7, 4, 2};
  auto ref = GetNodeFeatures(graph, "person", q, {"id"});
  auto got = GetNodeFeatures(graph, "person", q, {"id"}, &cache);
  REQUIRE(ref.status().ok());
  REQUIRE(got.status().ok());
  REQUIRE(ref.value()->Equals(*got.value()));
}

TEST_CASE_METHOD(GlobalFixture, "GetNodeFeatures static_cache all hit or all miss") {
  auto maybe_graph = GraphInfo::Load(test_data_dir + "/ldbc_sample/parquet/ldbc_sample.graph.yml");
  REQUIRE(maybe_graph.status().ok());
  auto graph = maybe_graph.value();

  StaticFeatureCache warm(graph);
  REQUIRE(warm.Pin("person", std::vector<IdType>({1, 2, 3}), {"id"}).ok());
  std::vector<IdType> inner = {3, 1, 2};
  auto r1 = GetNodeFeatures(graph, "person", inner, {"id"});
  auto g1 = GetNodeFeatures(graph, "person", inner, {"id"}, &warm);
  REQUIRE(r1.status().ok());
  REQUIRE(g1.status().ok());
  REQUIRE(r1.value()->Equals(*g1.value()));

  StaticFeatureCache cold(graph);
  std::vector<IdType> any = {7, 8};
  auto r2 = GetNodeFeatures(graph, "person", any, {"id"});
  auto g2 = GetNodeFeatures(graph, "person", any, {"id"}, &cold);
  REQUIRE(r2.status().ok());
  REQUIRE(g2.status().ok());
  REQUIRE(r2.value()->Equals(*g2.value()));
}

}  // namespace graphar::ml
