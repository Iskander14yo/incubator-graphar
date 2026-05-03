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

namespace graphar::ml {

class ChunkReadManager;
struct FeatureCursorStats;
struct SamplingResult;
struct SamplingBatchState;

struct EdgeSamplingPipelineOptions {
  size_t num_readers = 1;
  size_t trail_capacity_chunks = 0;
  size_t num_processors = 1;
  size_t max_active_batches = 1;
};

struct EdgeSamplingPipelineStats {
  uint64_t submitted_batches = 0;
  uint64_t completed_batches = 0;
  uint64_t active_batches_current = 0;
  uint64_t pending_batches_peak = 0;
  uint64_t active_offset_chunk_keys_current = 0;
  uint64_t active_offset_chunk_keys_peak = 0;
  uint64_t active_adj_chunk_keys_current = 0;
  uint64_t active_adj_chunk_keys_peak = 0;
  uint64_t read_queue_current = 0;
  uint64_t processor_queue_current = 0;
  uint64_t offset_chunk_subscriptions = 0;
  uint64_t offset_chunk_reads = 0;
  uint64_t offset_chunk_reuses = 0;
  uint64_t adj_chunk_subscriptions = 0;
  uint64_t adj_chunk_reads = 0;
  uint64_t adj_chunk_reuses = 0;
  uint64_t processor_tasks = 0;
  uint64_t processor_wait_ms_sum = 0;
  uint64_t processor_service_ms_sum = 0;
};

class SamplingBatchHandle {
 public:
  using BatchResult = Result<SamplingResult>;

  SamplingBatchHandle();
  explicit SamplingBatchHandle(std::shared_ptr<SamplingBatchState> state);
  ~SamplingBatchHandle();

  SamplingBatchHandle(const SamplingBatchHandle&);
  SamplingBatchHandle& operator=(const SamplingBatchHandle&);
  SamplingBatchHandle(SamplingBatchHandle&&) noexcept;
  SamplingBatchHandle& operator=(SamplingBatchHandle&&) noexcept;

  BatchResult Wait() const;
  uint64_t sampling_ms() const;
  bool valid() const;

  std::shared_ptr<SamplingBatchState> state_;
};

class EdgeSamplingPipelineCoordinator {
 public:
  explicit EdgeSamplingPipelineCoordinator(
      std::shared_ptr<ChunkReadManager> chunk_manager,
      EdgeSamplingPipelineOptions options = {});
  ~EdgeSamplingPipelineCoordinator();

  EdgeSamplingPipelineCoordinator(const EdgeSamplingPipelineCoordinator&);
  EdgeSamplingPipelineCoordinator& operator=(
      const EdgeSamplingPipelineCoordinator&);
  EdgeSamplingPipelineCoordinator(EdgeSamplingPipelineCoordinator&&) noexcept;
  EdgeSamplingPipelineCoordinator& operator=(
      EdgeSamplingPipelineCoordinator&&) noexcept;

  Result<std::shared_ptr<SamplingBatchHandle>> SubmitSeedBatch(
      const std::shared_ptr<GraphInfo>& graph_info,
      const std::string& vertex_type, const std::string& edge_type,
      const std::vector<IdType>& seed_nodes, const std::vector<int>& fanout,
      uint64_t seed);

  EdgeSamplingPipelineStats Stats() const;
  FeatureCursorStats CursorStats() const;
  void Shutdown();

 private:
  struct Impl;

  std::shared_ptr<Impl> impl_;
};

}  // namespace graphar::ml
