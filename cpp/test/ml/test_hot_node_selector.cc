#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "graphar/graph_info.h"
#include "graphar/ml/hot_node_selector.h"

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

}  // namespace

TEST_CASE_METHOD(GlobalFixture, "DegreeHotNodeSelector - Select prefers high in-degree") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  DegreeHotNodeSelector sel(graph_info, kVertexType, kEdgeType);

  auto r = sel.Select(10);
  REQUIRE(r.status().ok());
  const auto& ids = r.value();
  REQUIRE(ids.size() == 10);
  REQUIRE(std::set<IdType>(ids.begin(), ids.end()).size() == ids.size());

  auto r_all = sel.Select(1000000);
  REQUIRE(r_all.status().ok());
  const auto& all_order = r_all.value();
  REQUIRE(all_order.size() > 10u);
  REQUIRE(std::vector<IdType>(all_order.begin(), all_order.begin() + 10) == ids);
}

TEST_CASE_METHOD(GlobalFixture, "DegreeHotNodeSelector - Curve is monotone and ends at 1") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  DegreeHotNodeSelector sel(graph_info, kVertexType, kEdgeType);

  auto cr = sel.Curve();
  REQUIRE(cr.status().ok());
  const auto& curve = cr.value();
  REQUIRE(!curve.empty());
  for (size_t i = 1; i < curve.size(); ++i) {
    REQUIRE(curve[i].cumulative_hit_rate >= curve[i - 1].cumulative_hit_rate - 1e-9);
    REQUIRE(curve[i].k >= curve[i - 1].k);
  }
  REQUIRE(std::abs(curve.back().cumulative_hit_rate - 1.0) < 1e-6);
}

TEST_CASE_METHOD(GlobalFixture, "EstimateHitRate matches mass for DegreeHotNodeSelector") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  DegreeHotNodeSelector sel(graph_info, kVertexType, kEdgeType);

  auto er = EstimateHitRate(sel, 50);
  REQUIRE(er.status().ok());
  double y = er.value();
  REQUIRE(y >= 0.0);
  REQUIRE(y <= 1.0 + 1e-9);

  auto cr = sel.Curve();
  REQUIRE(cr.status().ok());
  auto interp = EstimateHitRate(sel, 17);
  REQUIRE(interp.status().ok());
  REQUIRE(interp.value() >= 0.0);
}

}  // namespace graphar::ml
