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
#include "graphar/ml/neighbor_sampling.h"
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
      .def_property_readonly("num_chunks", &graphar::ml::FeatureCache::num_chunks,
           "Number of chunks currently in the cache.")
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

  // Bind get_node_features function
  m.def("get_node_features",
      [](const std::shared_ptr<graphar::GraphInfo>& graph_info,
         const std::string& vertex_type,
         const std::vector<graphar::IdType>& node_ids,
         const std::vector<std::string>& properties,
         graphar::ml::FeatureCache* cache) {
        auto result = [&]() {
          py::gil_scoped_release release; // release GIL
          return graphar::ml::GetNodeFeatures(
              graph_info, vertex_type, node_ids, properties, cache);
        }();
        auto table = ThrowOrReturn(result);
        return table_to_pyarrow(table);
      },
      py::arg("graph_info"),
      py::arg("vertex_type"),
      py::arg("node_ids"),
      py::arg("properties"),
      py::arg("cache").none(true) = static_cast<graphar::ml::FeatureCache*>(nullptr),
      "Fetch node properties for given internal IDs");
}
