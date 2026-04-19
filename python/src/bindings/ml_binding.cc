/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "utils/pybind_util.h"

#include "arrow/api.h"
#include "arrow/c/bridge.h"
#include "graphar/ml/feature_cache.h"
#include "graphar/ml/hot_node_selector.h"
#include "graphar/ml/neighbor_sampling.h"
#include "graphar/ml/static_feature_cache.h"
#include "graphar/graph_info.h"

namespace py = pybind11;

namespace {

// Convert Arrow Table to PyArrow Table via C Data Interface
py::object table_to_pyarrow(const std::shared_ptr<arrow::Table>& table) {
  // Combine into single batch for simpler export
  auto batch_result = table->CombineChunksToBatch();
  if (!batch_result.ok()) {
    throw std::runtime_error("Failed to combine table chunks: " + 
                             batch_result.status().ToString());
  }
  auto batch = batch_result.ValueOrDie();

  // Export via C Data Interface
  ArrowArray c_array;
  ArrowSchema c_schema;
  auto status = arrow::ExportRecordBatch(*batch, &c_array, &c_schema);
  if (!status.ok()) {
    throw std::runtime_error("Failed to export batch: " + status.ToString());
  }

  // Import into PyArrow
  py::module pyarrow = py::module::import("pyarrow");
  py::object pa_batch_cls = pyarrow.attr("RecordBatch");
  auto import_fn = pa_batch_cls.attr("_import_from_c");

  intptr_t c_array_ptr = reinterpret_cast<intptr_t>(&c_array);
  intptr_t c_schema_ptr = reinterpret_cast<intptr_t>(&c_schema);

  py::object pa_batch = import_fn(c_array_ptr, c_schema_ptr);
  
  // Convert RecordBatch to Table
  return pyarrow.attr("Table").attr("from_batches")(py::make_tuple(pa_batch));
}

}  // namespace

extern "C" void bind_ml_api(pybind11::module_& m) {
  // Bind FeatureCache class
  py::class_<graphar::ml::FeatureCache>(m, "FeatureCache",
      "Thread-safe O(1) LFU cache for node feature chunks.")
      .def(py::init<size_t>(), py::arg("max_bytes"),
           "Create a cache with the given byte budget.")
      .def_property_readonly("hits", &graphar::ml::FeatureCache::hits,
           "Total cache hit count.")
      .def_property_readonly("misses", &graphar::ml::FeatureCache::misses,
           "Total cache miss count.")
      .def_property_readonly("hit_rate", &graphar::ml::FeatureCache::hit_rate,
           "hits / (hits + misses), or 0.0 if no lookups yet.")
      .def_property_readonly("chunks_read", &graphar::ml::FeatureCache::chunks_read,
           "Total property-group chunk reads performed (disk I/O calls).")
      .def_property_readonly("chunks_skipped", &graphar::ml::FeatureCache::chunks_skipped,
           "Chunk reads avoided because all needed nodes were in cache.")
      .def_property_readonly("io_saved_pct", &graphar::ml::FeatureCache::io_saved_pct,
           "chunks_skipped / (chunks_read + chunks_skipped) as a percentage.")
      .def_property_readonly("num_nodes", &graphar::ml::FeatureCache::num_nodes,
           "Number of node rows currently in the cache.")
      .def_property_readonly("size_mb",
           [](const graphar::ml::FeatureCache& c) {
             return static_cast<double>(c.size_bytes()) / (1024.0 * 1024.0);
           },
           "Current cache size in megabytes.")
      .def_property_readonly("max_size_mb",
           [](const graphar::ml::FeatureCache& c) {
             return static_cast<double>(c.max_bytes()) / (1024.0 * 1024.0);
           },
           "Cache byte budget in megabytes.")
      .def("clear", &graphar::ml::FeatureCache::Clear,
           "Remove all cached entries (stats are preserved).");

  // Bind SamplingResult struct
  py::class_<graphar::ml::SamplingResult>(m, "SamplingResult")
      .def(py::init<>())
      .def_readwrite("sampled_nodes", &graphar::ml::SamplingResult::sampled_nodes)
      .def_readwrite("src_indices", &graphar::ml::SamplingResult::src_indices)
      .def_readwrite("dst_indices", &graphar::ml::SamplingResult::dst_indices)
      .def_readwrite("num_sampled_nodes_per_hop",
                     &graphar::ml::SamplingResult::num_sampled_nodes_per_hop)
      .def_readwrite("num_sampled_edges_per_hop",
                     &graphar::ml::SamplingResult::num_sampled_edges_per_hop);

  // Bind sample_neighbors function
  m.def("sample_neighbors", 
      [](const std::shared_ptr<graphar::GraphInfo>& graph_info,
         const std::string& vertex_type,
         const std::string& edge_type,
         const std::vector<graphar::IdType>& seed_nodes,
         const std::vector<int>& fanout, uint64_t seed) {
        auto result = [&]() {
          py::gil_scoped_release release; // release GIL
          return graphar::ml::SampleNeighbors(
              graph_info, vertex_type, edge_type, seed_nodes, fanout, seed);
        }();
        return ThrowOrReturn(result);
      },
      py::arg("graph_info"),
      py::arg("vertex_type"),
      py::arg("edge_type"),
      py::arg("seed_nodes"),
      py::arg("fanout"),
      py::arg("seed"),
      "Sample multi-hop neighbors for given seed nodes");

  py::class_<graphar::ml::StaticFeatureCache>(
      m, "StaticFeatureCache",
      "Immutable columnar feature tier: Pin once, then lock-free Lookup.")
      .def(py::init<std::shared_ptr<graphar::GraphInfo>>(), py::arg("graph_info"))
      .def(
          "pin",
          [](graphar::ml::StaticFeatureCache& self, const std::string& vertex_type,
             const std::vector<graphar::IdType>& node_ids,
             const std::vector<std::string>& properties) {
            CheckStatus(self.Pin(vertex_type, node_ids, properties));
          },
          py::arg("vertex_type"), py::arg("node_ids"), py::arg("properties"))
      .def_property_readonly("num_nodes", &graphar::ml::StaticFeatureCache::num_nodes)
      .def_property_readonly("size_bytes", &graphar::ml::StaticFeatureCache::size_bytes)
      .def_property_readonly("hits", &graphar::ml::StaticFeatureCache::hits)
      .def_property_readonly("misses", &graphar::ml::StaticFeatureCache::misses)
      .def_property_readonly("hit_rate", &graphar::ml::StaticFeatureCache::hit_rate);

  py::class_<graphar::ml::DegreeHotNodeSelector>(m, "DegreeHotNodeSelector")
      .def(py::init<std::shared_ptr<graphar::GraphInfo>, std::string, std::string>(),
           py::arg("graph_info"), py::arg("vertex_type"), py::arg("edge_type"))
      .def(
          "select",
          [](graphar::ml::DegreeHotNodeSelector& self, size_t top_k) {
            return ThrowOrReturn(self.Select(top_k));
          },
          py::arg("top_k"))
      .def(
          "curve",
          [](graphar::ml::DegreeHotNodeSelector& self) {
            auto cv = ThrowOrReturn(self.Curve());
            py::list out;
            for (const auto& p : cv) {
              out.append(py::make_tuple(p.k, p.cumulative_hit_rate));
            }
            return out;
          });

  m.def(
      "estimate_hit_rate",
      [](graphar::ml::DegreeHotNodeSelector& sel, size_t top_k) {
        return ThrowOrReturn(graphar::ml::EstimateHitRate(
            static_cast<graphar::ml::HotNodeSelector&>(sel), top_k));
      },
      py::arg("selector"), py::arg("top_k"));

  m.def(
      "get_node_features",
      [](const std::shared_ptr<graphar::GraphInfo>& graph_info,
         const std::string& vertex_type,
         const std::vector<graphar::IdType>& node_ids,
         const std::vector<std::string>& properties, py::object cache_arg,
         py::object static_cache_arg) {
        if (!cache_arg.is_none() && !static_cache_arg.is_none()) {
          throw std::runtime_error("pass only one of cache and static_cache");
        }
        std::shared_ptr<arrow::Table> table;
        if (!static_cache_arg.is_none()) {
          auto* sc = static_cache_arg.cast<graphar::ml::StaticFeatureCache*>();
          auto result = [&]() {
            py::gil_scoped_release release;
            return graphar::ml::GetNodeFeatures(graph_info, vertex_type, node_ids,
                                                properties, sc);
          }();
          table = ThrowOrReturn(result);
        } else {
          graphar::ml::FeatureCache* fc = nullptr;
          if (!cache_arg.is_none()) {
            fc = cache_arg.cast<graphar::ml::FeatureCache*>();
          }
          auto result = [&]() {
            py::gil_scoped_release release;
            return graphar::ml::GetNodeFeatures(graph_info, vertex_type, node_ids,
                                                properties, fc);
          }();
          table = ThrowOrReturn(result);
        }
        return table_to_pyarrow(table);
      },
      py::arg("graph_info"), py::arg("vertex_type"), py::arg("node_ids"),
      py::arg("properties"), py::arg("cache").none(true) = py::none(),
      py::arg("static_cache").none(true) = py::none(),
      "Fetch node properties for given internal IDs");
}
