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

#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "graphar/fwd.h"
#include "graphar/result.h"

namespace arrow {
class Table;
}

namespace graphar::ml {

enum class ChunkReadKind {
  kVertexProperty,
  kEdgeOffset,
  kEdgeAdjList,
};

struct ChunkReadKey {
  ChunkReadKind kind = ChunkReadKind::kVertexProperty;
  std::string graph_prefix;
  std::string vertex_type;
  std::string edge_type;
  std::string dst_type;
  std::string property_group_prefix;
  AdjListType adj_list_type = static_cast<AdjListType>(0);
  FileType file_type = FileType::PARQUET;
  IdType vertex_chunk_id = 0;
  IdType chunk_id = 0;

  bool operator==(const ChunkReadKey& other) const;
};

struct ChunkReadKeyHash {
  size_t operator()(const ChunkReadKey& key) const;
};

struct ChunkReadManagerOptions {
  bool enable_singleflight = true;
  size_t ram_budget_bytes = 0;
};

struct ChunkReadStats {
  uint64_t requests = 0;
  uint64_t leaders = 0;
  uint64_t waiters = 0;
  uint64_t completed = 0;
  uint64_t failed = 0;
  uint64_t ram_cache_hits = 0;
  uint64_t ram_cache_misses = 0;
  uint64_t ram_cache_evictions = 0;
  uint64_t ram_cache_bytes = 0;
};

class ChunkReadManager {
 public:
  using TablePtr = std::shared_ptr<arrow::Table>;
  using TableResult = Result<TablePtr>;
  using TableLoader = std::function<TableResult()>;

  explicit ChunkReadManager(ChunkReadManagerOptions options = {});

  TableResult GetOrLoad(const ChunkReadKey& key, const TableLoader& loader);

  TableResult GetVertexPropertyChunk(
      const std::shared_ptr<GraphInfo>& graph_info,
      const std::string& vertex_type,
      const std::shared_ptr<PropertyGroup>& property_group, IdType chunk_id);

  TableResult GetEdgeOffsetChunk(const std::shared_ptr<GraphInfo>& graph_info,
                                 const std::string& src_type,
                                 const std::string& edge_type,
                                 const std::string& dst_type,
                                 AdjListType adj_list_type,
                                 IdType vertex_chunk_id);

  TableResult GetEdgeAdjListChunk(const std::shared_ptr<GraphInfo>& graph_info,
                                  const std::string& src_type,
                                  const std::string& edge_type,
                                  const std::string& dst_type,
                                  AdjListType adj_list_type,
                                  IdType vertex_chunk_id, IdType chunk_id);

  ChunkReadStats stats() const;

 private:
  struct CacheEntry {
    TablePtr table;
    size_t bytes = 0;
    std::list<ChunkReadKey>::iterator lru_it;
  };

  void RecordResult(const TableResult& result);
  TablePtr LookupRamCacheLocked(const ChunkReadKey& key);
  void InsertRamCacheLocked(const ChunkReadKey& key, const TablePtr& table);
  void EvictRamCacheLocked(size_t bytes_needed);

  ChunkReadManagerOptions options_;
  mutable std::mutex mutex_;
  std::list<ChunkReadKey> ram_cache_lru_;
  std::unordered_map<ChunkReadKey, CacheEntry, ChunkReadKeyHash> ram_cache_;
  size_t ram_cache_bytes_ = 0;
  std::unordered_map<ChunkReadKey, std::shared_future<TableResult>,
                     ChunkReadKeyHash>
      in_flight_;

  std::atomic<uint64_t> requests_{0};
  std::atomic<uint64_t> leaders_{0};
  std::atomic<uint64_t> waiters_{0};
  std::atomic<uint64_t> completed_{0};
  std::atomic<uint64_t> failed_{0};
  std::atomic<uint64_t> ram_cache_hits_{0};
  std::atomic<uint64_t> ram_cache_misses_{0};
  std::atomic<uint64_t> ram_cache_evictions_{0};
};

}  // namespace graphar::ml
