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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "graphar/fwd.h"
#include "graphar/result.h"

namespace arrow {
class Table;
}

namespace graphar::ml {

class ChunkReadManager;

struct FeaturePipelineOptions {
  size_t num_readers = 1;
  size_t num_stitchers = 1;
  size_t max_active_batches = 1;
  size_t max_queued_chunk_reads = 1;
  size_t max_queued_stitch_tasks = 1;
};

struct FeaturePipelineStats {
  uint64_t submitted_batches = 0;
  uint64_t completed_batches = 0;
  uint64_t pending_batches_peak = 0;
  uint64_t active_chunk_keys_peak = 0;
  uint64_t chunk_subscriptions = 0;
  uint64_t chunk_reads = 0;
  uint64_t chunk_reuses = 0;
  uint64_t stitch_tasks = 0;
  uint64_t stitch_wait_ms_sum = 0;
  uint64_t stitch_service_ms_sum = 0;
};

struct FeatureBatchState;

class FeatureBatchHandle {
 public:
  using TablePtr = std::shared_ptr<arrow::Table>;
  using TableResult = Result<TablePtr>;

  FeatureBatchHandle();
  explicit FeatureBatchHandle(std::shared_ptr<FeatureBatchState> state);
  ~FeatureBatchHandle();

  FeatureBatchHandle(const FeatureBatchHandle&);
  FeatureBatchHandle& operator=(const FeatureBatchHandle&);
  FeatureBatchHandle(FeatureBatchHandle&&) noexcept;
  FeatureBatchHandle& operator=(FeatureBatchHandle&&) noexcept;

  TableResult Wait() const;
  uint64_t feature_fetch_ms() const;
  bool valid() const;

  std::shared_ptr<FeatureBatchState> state_;
};

class FeaturePipelineCoordinator {
 public:
  explicit FeaturePipelineCoordinator(
      std::shared_ptr<ChunkReadManager> chunk_manager,
      FeaturePipelineOptions options = {});
  ~FeaturePipelineCoordinator();

  FeaturePipelineCoordinator(const FeaturePipelineCoordinator&);
  FeaturePipelineCoordinator& operator=(const FeaturePipelineCoordinator&);
  FeaturePipelineCoordinator(FeaturePipelineCoordinator&&) noexcept;
  FeaturePipelineCoordinator& operator=(FeaturePipelineCoordinator&&) noexcept;

  Result<std::shared_ptr<FeatureBatchHandle>> SubmitSampledBatch(
      const std::shared_ptr<GraphInfo>& graph_info,
      const std::string& vertex_type, const std::vector<IdType>& node_ids,
      const std::vector<std::string>& properties);

  FeaturePipelineStats Stats() const;
  void Shutdown();

 private:
  struct Impl;

  std::shared_ptr<Impl> impl_;
};

}  // namespace graphar::ml
