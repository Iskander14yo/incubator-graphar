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
#include "graphar/ml/chunk_read_manager.h"
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

  py::class_<graphar::ml::ChunkReadManagerOptions>(
      m, "_ChunkReadManagerOptions")
      .def(py::init<>())
      .def_readwrite("enable_singleflight",
                     &graphar::ml::ChunkReadManagerOptions::enable_singleflight)
      .def_readwrite("ram_budget_bytes",
                     &graphar::ml::ChunkReadManagerOptions::ram_budget_bytes);

  py::class_<graphar::ml::ChunkReadStats>(m, "_ChunkReadStats")
      .def_readonly("requests", &graphar::ml::ChunkReadStats::requests)
      .def_readonly("leaders", &graphar::ml::ChunkReadStats::leaders)
      .def_readonly("waiters", &graphar::ml::ChunkReadStats::waiters)
      .def_readonly("completed", &graphar::ml::ChunkReadStats::completed)
      .def_readonly("failed", &graphar::ml::ChunkReadStats::failed)
      .def_readonly("ram_cache_hits",
                    &graphar::ml::ChunkReadStats::ram_cache_hits)
      .def_readonly("ram_cache_misses",
                    &graphar::ml::ChunkReadStats::ram_cache_misses)
      .def_readonly("ram_cache_evictions",
                    &graphar::ml::ChunkReadStats::ram_cache_evictions)
      .def_readonly("ram_cache_bytes",
                    &graphar::ml::ChunkReadStats::ram_cache_bytes);

  py::class_<graphar::ml::ChunkReadManager,
             std::shared_ptr<graphar::ml::ChunkReadManager>>(
      m, "_ChunkReadManager")
      .def(py::init<graphar::ml::ChunkReadManagerOptions>(),
           py::arg("options") = graphar::ml::ChunkReadManagerOptions{})
      .def("stats", &graphar::ml::ChunkReadManager::stats);

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
         const std::shared_ptr<graphar::ml::ChunkReadManager>& chunk_manager) {
        auto result = [&]() {
          py::gil_scoped_release release; // release GIL
          return graphar::ml::GetNodeFeatures(
              graph_info, vertex_type, node_ids, properties,
              chunk_manager.get());
        }();
        auto table = ThrowOrReturn(result);
        return table_to_pyarrow(table);
      },
      py::arg("graph_info"),
      py::arg("vertex_type"),
      py::arg("node_ids"),
      py::arg("properties"),
      py::arg("chunk_manager") = nullptr,
      "Fetch node properties for given internal IDs");
}
