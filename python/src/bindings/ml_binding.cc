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
#include "graphar/graph_info.h"
#include "graphar/ml/chunk_read_manager.h"
#include "graphar/ml/edge_pipeline.h"
#include "graphar/ml/feature_pipeline.h"
#include "graphar/ml/neighbor_sampling.h"

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
      .def_readwrite("sampled_nodes",
                     &graphar::ml::SamplingResult::sampled_nodes)
      .def_readwrite("src_indices", &graphar::ml::SamplingResult::src_indices)
      .def_readwrite("dst_indices", &graphar::ml::SamplingResult::dst_indices)
      .def_readwrite("num_sampled_nodes_per_hop",
                     &graphar::ml::SamplingResult::num_sampled_nodes_per_hop)
      .def_readwrite("num_sampled_edges_per_hop",
                     &graphar::ml::SamplingResult::num_sampled_edges_per_hop);

  py::class_<graphar::ml::ChunkReadManagerOptions>(m,
                                                   "_ChunkReadManagerOptions")
      .def(py::init<>())
      .def_readwrite("enable_singleflight",
                     &graphar::ml::ChunkReadManagerOptions::enable_singleflight)
      .def_readwrite("ram_budget_bytes",
                     &graphar::ml::ChunkReadManagerOptions::ram_budget_bytes)
      .def_readwrite("edge_offset_ram_budget_bytes",
                     &graphar::ml::ChunkReadManagerOptions::
                         edge_offset_ram_budget_bytes)
      .def_readwrite("edge_adj_list_ram_budget_bytes",
                     &graphar::ml::ChunkReadManagerOptions::
                         edge_adj_list_ram_budget_bytes)
      .def_readwrite("feature_cursor_count",
                     &graphar::ml::ChunkReadManagerOptions::feature_cursor_count)
      .def_readwrite(
          "feature_cursor_trail_capacity_chunks",
          &graphar::ml::ChunkReadManagerOptions::
              feature_cursor_trail_capacity_chunks);

  py::class_<graphar::ml::ChunkReadStats>(m, "_ChunkReadStats")
      .def_readonly("requests", &graphar::ml::ChunkReadStats::requests)
      .def_readonly("leaders", &graphar::ml::ChunkReadStats::leaders)
      .def_readonly("waiters", &graphar::ml::ChunkReadStats::waiters)
      .def_readonly("completed", &graphar::ml::ChunkReadStats::completed)
      .def_readonly("failed", &graphar::ml::ChunkReadStats::failed)
      .def_readonly("vertex_property_requests",
                    &graphar::ml::ChunkReadStats::vertex_property_requests)
      .def_readonly("vertex_property_leaders",
                    &graphar::ml::ChunkReadStats::vertex_property_leaders)
      .def_readonly("vertex_property_waiters",
                    &graphar::ml::ChunkReadStats::vertex_property_waiters)
      .def_readonly("vertex_property_completed",
                    &graphar::ml::ChunkReadStats::vertex_property_completed)
      .def_readonly("vertex_property_failed",
                    &graphar::ml::ChunkReadStats::vertex_property_failed)
      .def_readonly("ram_cache_hits",
                    &graphar::ml::ChunkReadStats::ram_cache_hits)
      .def_readonly("ram_cache_misses",
                    &graphar::ml::ChunkReadStats::ram_cache_misses)
      .def_readonly("ram_cache_evictions",
                    &graphar::ml::ChunkReadStats::ram_cache_evictions)
      .def_readonly("ram_cache_bytes",
                    &graphar::ml::ChunkReadStats::ram_cache_bytes)
      .def_readonly("edge_offset_requests",
                    &graphar::ml::ChunkReadStats::edge_offset_requests)
      .def_readonly("edge_offset_leaders",
                    &graphar::ml::ChunkReadStats::edge_offset_leaders)
      .def_readonly("edge_offset_waiters",
                    &graphar::ml::ChunkReadStats::edge_offset_waiters)
      .def_readonly("edge_offset_completed",
                    &graphar::ml::ChunkReadStats::edge_offset_completed)
      .def_readonly("edge_offset_failed",
                    &graphar::ml::ChunkReadStats::edge_offset_failed)
      .def_readonly("vertex_property_ram_cache_hits",
                    &graphar::ml::ChunkReadStats::
                        vertex_property_ram_cache_hits)
      .def_readonly("vertex_property_ram_cache_misses",
                    &graphar::ml::ChunkReadStats::
                        vertex_property_ram_cache_misses)
      .def_readonly("vertex_property_ram_cache_evictions",
                    &graphar::ml::ChunkReadStats::
                        vertex_property_ram_cache_evictions)
      .def_readonly("vertex_property_ram_cache_bytes",
                    &graphar::ml::ChunkReadStats::
                        vertex_property_ram_cache_bytes)
      .def_readonly("edge_offset_ram_cache_hits",
                    &graphar::ml::ChunkReadStats::edge_offset_ram_cache_hits)
      .def_readonly("edge_offset_ram_cache_misses",
                    &graphar::ml::ChunkReadStats::
                        edge_offset_ram_cache_misses)
      .def_readonly("edge_offset_ram_cache_evictions",
                    &graphar::ml::ChunkReadStats::
                        edge_offset_ram_cache_evictions)
      .def_readonly("edge_offset_ram_cache_bytes",
                    &graphar::ml::ChunkReadStats::edge_offset_ram_cache_bytes)
      .def_readonly("edge_adj_list_requests",
                    &graphar::ml::ChunkReadStats::edge_adj_list_requests)
      .def_readonly("edge_adj_list_leaders",
                    &graphar::ml::ChunkReadStats::edge_adj_list_leaders)
      .def_readonly("edge_adj_list_waiters",
                    &graphar::ml::ChunkReadStats::edge_adj_list_waiters)
      .def_readonly("edge_adj_list_completed",
                    &graphar::ml::ChunkReadStats::edge_adj_list_completed)
      .def_readonly("edge_adj_list_failed",
                    &graphar::ml::ChunkReadStats::edge_adj_list_failed)
      .def_readonly("edge_adj_list_ram_cache_hits",
                    &graphar::ml::ChunkReadStats::
                        edge_adj_list_ram_cache_hits)
      .def_readonly("edge_adj_list_ram_cache_misses",
                    &graphar::ml::ChunkReadStats::
                        edge_adj_list_ram_cache_misses)
      .def_readonly("edge_adj_list_ram_cache_evictions",
                    &graphar::ml::ChunkReadStats::
                        edge_adj_list_ram_cache_evictions)
      .def_readonly("edge_adj_list_ram_cache_bytes",
                    &graphar::ml::ChunkReadStats::
                        edge_adj_list_ram_cache_bytes);

  py::class_<graphar::ml::ChunkReadManager,
             std::shared_ptr<graphar::ml::ChunkReadManager>>(
      m, "_ChunkReadManager")
      .def(py::init<graphar::ml::ChunkReadManagerOptions>(),
           py::arg("options") = graphar::ml::ChunkReadManagerOptions{})
      .def("stats", &graphar::ml::ChunkReadManager::stats)
      .def("feature_cursor_stats",
           &graphar::ml::ChunkReadManager::feature_cursor_stats)
      .def("shutdown", &graphar::ml::ChunkReadManager::Shutdown);

  py::class_<graphar::ml::FeaturePipelineOptions>(m,
                                                  "_FeaturePipelineOptions")
      .def(py::init<>())
      .def_readwrite("num_readers",
                     &graphar::ml::FeaturePipelineOptions::num_readers)
      .def_readwrite("num_stitchers",
                     &graphar::ml::FeaturePipelineOptions::num_stitchers)
      .def_readwrite(
          "max_active_batches",
          &graphar::ml::FeaturePipelineOptions::max_active_batches)
      .def_readwrite(
          "max_queued_stitch_tasks",
          &graphar::ml::FeaturePipelineOptions::max_queued_stitch_tasks);

  py::class_<graphar::ml::FeaturePipelineStats>(m, "_FeaturePipelineStats")
      .def_readonly("submitted_batches",
                    &graphar::ml::FeaturePipelineStats::submitted_batches)
      .def_readonly("completed_batches",
                    &graphar::ml::FeaturePipelineStats::completed_batches)
      .def_readonly("active_batches_current",
                    &graphar::ml::FeaturePipelineStats::active_batches_current)
      .def_readonly("pending_batches_peak",
                    &graphar::ml::FeaturePipelineStats::pending_batches_peak)
      .def_readonly("active_chunk_keys_current",
                    &graphar::ml::FeaturePipelineStats::active_chunk_keys_current)
      .def_readonly("active_chunk_keys_peak",
                    &graphar::ml::FeaturePipelineStats::active_chunk_keys_peak)
      .def_readonly("read_queue_current",
                    &graphar::ml::FeaturePipelineStats::read_queue_current)
      .def_readonly("stitch_queue_current",
                    &graphar::ml::FeaturePipelineStats::stitch_queue_current)
      .def_readonly("chunk_subscriptions",
                    &graphar::ml::FeaturePipelineStats::chunk_subscriptions)
      .def_readonly("chunk_reads",
                    &graphar::ml::FeaturePipelineStats::chunk_reads)
      .def_readonly("chunk_reuses",
                    &graphar::ml::FeaturePipelineStats::chunk_reuses)
      .def_readonly("stitch_tasks",
                    &graphar::ml::FeaturePipelineStats::stitch_tasks)
      .def_readonly(
          "stitch_wait_ms_sum",
          &graphar::ml::FeaturePipelineStats::stitch_wait_ms_sum)
      .def_readonly(
          "stitch_service_ms_sum",
          &graphar::ml::FeaturePipelineStats::stitch_service_ms_sum);

  py::class_<graphar::ml::EdgeSamplingPipelineOptions>(
      m, "_EdgeSamplingPipelineOptions")
      .def(py::init<>())
      .def_readwrite("num_readers",
                     &graphar::ml::EdgeSamplingPipelineOptions::num_readers)
      .def_readwrite(
          "trail_capacity_chunks",
          &graphar::ml::EdgeSamplingPipelineOptions::trail_capacity_chunks)
      .def_readwrite("num_processors",
                     &graphar::ml::EdgeSamplingPipelineOptions::num_processors)
      .def_readwrite(
          "max_active_batches",
          &graphar::ml::EdgeSamplingPipelineOptions::max_active_batches);

  py::class_<graphar::ml::EdgeSamplingPipelineStats>(
      m, "_EdgeSamplingPipelineStats")
      .def_readonly("submitted_batches",
                    &graphar::ml::EdgeSamplingPipelineStats::submitted_batches)
      .def_readonly("completed_batches",
                    &graphar::ml::EdgeSamplingPipelineStats::completed_batches)
      .def_readonly(
          "active_batches_current",
          &graphar::ml::EdgeSamplingPipelineStats::active_batches_current)
      .def_readonly(
          "pending_batches_peak",
          &graphar::ml::EdgeSamplingPipelineStats::pending_batches_peak)
      .def_readonly("active_offset_chunk_keys_current",
                    &graphar::ml::EdgeSamplingPipelineStats::
                        active_offset_chunk_keys_current)
      .def_readonly("active_offset_chunk_keys_peak",
                    &graphar::ml::EdgeSamplingPipelineStats::
                        active_offset_chunk_keys_peak)
      .def_readonly("active_adj_chunk_keys_current",
                    &graphar::ml::EdgeSamplingPipelineStats::
                        active_adj_chunk_keys_current)
      .def_readonly("active_adj_chunk_keys_peak",
                    &graphar::ml::EdgeSamplingPipelineStats::
                        active_adj_chunk_keys_peak)
      .def_readonly("read_queue_current",
                    &graphar::ml::EdgeSamplingPipelineStats::read_queue_current)
      .def_readonly("processor_queue_current",
                    &graphar::ml::EdgeSamplingPipelineStats::
                        processor_queue_current)
      .def_readonly("offset_chunk_subscriptions",
                    &graphar::ml::EdgeSamplingPipelineStats::
                        offset_chunk_subscriptions)
      .def_readonly("offset_chunk_reads",
                    &graphar::ml::EdgeSamplingPipelineStats::
                        offset_chunk_reads)
      .def_readonly("offset_chunk_reuses",
                    &graphar::ml::EdgeSamplingPipelineStats::
                        offset_chunk_reuses)
      .def_readonly("adj_chunk_subscriptions",
                    &graphar::ml::EdgeSamplingPipelineStats::
                        adj_chunk_subscriptions)
      .def_readonly("adj_chunk_reads",
                    &graphar::ml::EdgeSamplingPipelineStats::adj_chunk_reads)
      .def_readonly("adj_chunk_reuses",
                    &graphar::ml::EdgeSamplingPipelineStats::adj_chunk_reuses)
      .def_readonly("processor_tasks",
                    &graphar::ml::EdgeSamplingPipelineStats::processor_tasks)
      .def_readonly("processor_wait_ms_sum",
                    &graphar::ml::EdgeSamplingPipelineStats::
                        processor_wait_ms_sum)
      .def_readonly("processor_service_ms_sum",
                    &graphar::ml::EdgeSamplingPipelineStats::
                        processor_service_ms_sum);

  py::class_<graphar::ml::FeatureBatchHandle,
             std::shared_ptr<graphar::ml::FeatureBatchHandle>>(
      m, "_FeatureBatchHandle")
      .def("wait", [](const graphar::ml::FeatureBatchHandle& handle) {
        auto result = [&]() {
          py::gil_scoped_release release;
          return handle.Wait();
        }();
        auto table = ThrowOrReturn(result);
        return table_to_pyarrow(table);
      })
      .def("feature_fetch_ms",
           &graphar::ml::FeatureBatchHandle::feature_fetch_ms)
      .def("valid", &graphar::ml::FeatureBatchHandle::valid);

  py::class_<graphar::ml::SamplingBatchHandle,
             std::shared_ptr<graphar::ml::SamplingBatchHandle>>(
      m, "_SamplingBatchHandle")
      .def("wait", [](const graphar::ml::SamplingBatchHandle& handle) {
        auto result = [&]() {
          py::gil_scoped_release release;
          return handle.Wait();
        }();
        return ThrowOrReturn(result);
      })
      .def("sampling_ms",
           &graphar::ml::SamplingBatchHandle::sampling_ms)
      .def("valid", &graphar::ml::SamplingBatchHandle::valid);

  py::class_<graphar::ml::FeaturePipelineCoordinator,
             std::shared_ptr<graphar::ml::FeaturePipelineCoordinator>>(
      m, "_FeaturePipelineCoordinator")
      .def(py::init<std::shared_ptr<graphar::ml::ChunkReadManager>,
                    graphar::ml::FeaturePipelineOptions>(),
           py::arg("chunk_manager"),
           py::arg("options") = graphar::ml::FeaturePipelineOptions{})
      .def(
          "submit_sampled_batch",
          [](graphar::ml::FeaturePipelineCoordinator& coordinator,
             const std::shared_ptr<graphar::GraphInfo>& graph_info,
             const std::string& vertex_type,
             const std::vector<graphar::IdType>& node_ids,
             const std::vector<std::string>& properties) {
            auto result = [&]() {
              py::gil_scoped_release release;
              return coordinator.SubmitSampledBatch(graph_info, vertex_type,
                                                    node_ids, properties);
            }();
            return ThrowOrReturn(result);
          },
          py::arg("graph_info"), py::arg("vertex_type"), py::arg("node_ids"),
          py::arg("properties"))
      .def("stats", &graphar::ml::FeaturePipelineCoordinator::Stats)
      .def("shutdown", &graphar::ml::FeaturePipelineCoordinator::Shutdown);

  py::class_<graphar::ml::EdgeSamplingPipelineCoordinator,
             std::shared_ptr<graphar::ml::EdgeSamplingPipelineCoordinator>>(
      m, "_EdgeSamplingPipelineCoordinator")
      .def(py::init<std::shared_ptr<graphar::ml::ChunkReadManager>,
                    graphar::ml::EdgeSamplingPipelineOptions>(),
           py::arg("chunk_manager"),
           py::arg("options") = graphar::ml::EdgeSamplingPipelineOptions{})
      .def(
          "submit_seed_batch",
          [](graphar::ml::EdgeSamplingPipelineCoordinator& coordinator,
             const std::shared_ptr<graphar::GraphInfo>& graph_info,
             const std::string& vertex_type, const std::string& edge_type,
             const std::vector<graphar::IdType>& seed_nodes,
             const std::vector<int>& fanout, uint64_t seed) {
            auto result = [&]() {
              py::gil_scoped_release release;
              return coordinator.SubmitSeedBatch(graph_info, vertex_type,
                                                 edge_type, seed_nodes, fanout,
                                                 seed);
            }();
            return ThrowOrReturn(result);
          },
          py::arg("graph_info"), py::arg("vertex_type"),
          py::arg("edge_type"), py::arg("seed_nodes"), py::arg("fanout"),
          py::arg("seed"))
      .def("stats", &graphar::ml::EdgeSamplingPipelineCoordinator::Stats)
      .def("cursor_stats",
           &graphar::ml::EdgeSamplingPipelineCoordinator::CursorStats)
      .def("shutdown", &graphar::ml::EdgeSamplingPipelineCoordinator::Shutdown);

  py::class_<graphar::ml::FeatureCursorStats>(m, "_FeatureCursorStats")
      .def_readonly("cursor_count",
                    &graphar::ml::FeatureCursorStats::cursor_count)
      .def_readonly("trail_capacity_chunks",
                    &graphar::ml::FeatureCursorStats::trail_capacity_chunks)
      .def_readonly("requests", &graphar::ml::FeatureCursorStats::requests)
      .def_readonly("requests_completed",
                    &graphar::ml::FeatureCursorStats::requests_completed)
      .def_readonly("requests_failed",
                    &graphar::ml::FeatureCursorStats::requests_failed)
      .def_readonly("active_requests_peak",
                    &graphar::ml::FeatureCursorStats::active_requests_peak)
      .def_readonly("chunks_read",
                    &graphar::ml::FeatureCursorStats::chunks_read)
      .def_readonly("chunks_served",
                    &graphar::ml::FeatureCursorStats::chunks_served)
      .def_readonly("chunk_order_wraps",
                    &graphar::ml::FeatureCursorStats::chunk_order_wraps)
      .def_readonly("rows_served",
                    &graphar::ml::FeatureCursorStats::rows_served)
      .def_readonly("batches_served",
                    &graphar::ml::FeatureCursorStats::batches_served)
      .def_readonly("trail_hits", &graphar::ml::FeatureCursorStats::trail_hits)
      .def_readonly("trail_misses",
                    &graphar::ml::FeatureCursorStats::trail_misses)
      .def_readonly("trail_evictions",
                    &graphar::ml::FeatureCursorStats::trail_evictions)
      .def_readonly("wait_ms_sum",
                    &graphar::ml::FeatureCursorStats::wait_ms_sum)
      .def_readonly("wait_ms_max",
                    &graphar::ml::FeatureCursorStats::wait_ms_max)
      .def_readonly("service_ms_sum",
                    &graphar::ml::FeatureCursorStats::service_ms_sum)
      .def_readonly("service_ms_max",
                    &graphar::ml::FeatureCursorStats::service_ms_max);

  // Bind sample_neighbors function
  m.def(
      "sample_neighbors",
      [](const std::shared_ptr<graphar::GraphInfo>& graph_info,
         const std::string& vertex_type, const std::string& edge_type,
         const std::vector<graphar::IdType>& seed_nodes,
         const std::vector<int>& fanout, uint64_t seed,
         const std::shared_ptr<graphar::ml::ChunkReadManager>& chunk_manager) {
        auto result = [&]() {
          py::gil_scoped_release release;  // release GIL
          return graphar::ml::SampleNeighbors(graph_info, vertex_type,
                                              edge_type, seed_nodes, fanout,
                                              seed, chunk_manager.get());
        }();
        return ThrowOrReturn(result);
      },
      py::arg("graph_info"), py::arg("vertex_type"), py::arg("edge_type"),
      py::arg("seed_nodes"), py::arg("fanout"), py::arg("seed"),
      py::arg("chunk_manager") = nullptr,
      "Sample multi-hop neighbors for given seed nodes");

  // Bind get_node_features function
  m.def(
      "get_node_features",
      [](const std::shared_ptr<graphar::GraphInfo>& graph_info,
         const std::string& vertex_type,
         const std::vector<graphar::IdType>& node_ids,
         const std::vector<std::string>& properties,
         const std::shared_ptr<graphar::ml::ChunkReadManager>& chunk_manager) {
        auto result = [&]() {
          py::gil_scoped_release release;  // release GIL
          return graphar::ml::GetNodeFeatures(graph_info, vertex_type, node_ids,
                                              properties, chunk_manager.get());
        }();
        auto table = ThrowOrReturn(result);
        return table_to_pyarrow(table);
      },
      py::arg("graph_info"), py::arg("vertex_type"), py::arg("node_ids"),
      py::arg("properties"), py::arg("chunk_manager") = nullptr,
      "Fetch node properties for given internal IDs");
}
