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
#include <future>
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

struct FeatureCursorOptions {
  size_t cursor_count = 1;
  size_t trail_capacity_chunks = 10;
};

struct FeatureCursorStats {
  uint64_t cursor_count = 0;
  uint64_t trail_capacity_chunks = 0;
  uint64_t requests = 0;
  uint64_t requests_completed = 0;
  uint64_t requests_failed = 0;
  uint64_t active_requests_peak = 0;
  uint64_t chunks_read = 0;
  uint64_t chunks_served = 0;
  uint64_t rows_served = 0;
  uint64_t batches_served = 0;
  uint64_t trail_hits = 0;
  uint64_t trail_misses = 0;
  uint64_t trail_evictions = 0;
  uint64_t wait_ms_sum = 0;
  uint64_t wait_ms_max = 0;
  uint64_t service_ms_sum = 0;
  uint64_t service_ms_max = 0;
};

class FeatureRequestHandle {
 public:
  using TablePtr = std::shared_ptr<arrow::Table>;
  using TableResult = Result<TablePtr>;

  explicit FeatureRequestHandle(std::shared_future<TableResult> future);

  TableResult Wait() const;

 private:
  std::shared_future<TableResult> future_;
};

class FeatureScanCoordinator {
 public:
  using HandlePtr = std::shared_ptr<FeatureRequestHandle>;
  using HandleResult = Result<HandlePtr>;

  FeatureScanCoordinator(std::shared_ptr<GraphInfo> graph_info,
                         std::shared_ptr<ChunkReadManager> chunk_manager,
                         FeatureCursorOptions options = {});
  ~FeatureScanCoordinator();

  FeatureScanCoordinator(const FeatureScanCoordinator&) = delete;
  FeatureScanCoordinator& operator=(const FeatureScanCoordinator&) = delete;

  HandleResult Submit(const std::string& vertex_type,
                      const std::vector<IdType>& node_ids,
                      const std::vector<std::string>& properties);
  FeatureCursorStats stats() const;
  void Shutdown();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace graphar::ml
