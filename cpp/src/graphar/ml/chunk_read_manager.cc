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

#include "graphar/ml/chunk_read_manager.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <iterator>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "graphar/arrow/chunk_reader.h"
#include "graphar/filesystem.h"
#include "graphar/graph_info.h"
#include "graphar/status.h"

namespace graphar::ml {
namespace {

template <typename T>
void HashCombine(size_t* seed, const T& value) {
  std::hash<T> hasher;
  *seed ^= hasher(value) + 0x9e3779b97f4a7c15ULL + (*seed << 6) + (*seed >> 2);
}

template <typename T>
void AtomicMax(std::atomic<T>* target, T value) {
  T current = target->load(std::memory_order_relaxed);
  while (current < value &&
         !target->compare_exchange_weak(current, value,
                                        std::memory_order_relaxed)) {
  }
}

uint64_t MillisecondsSince(std::chrono::steady_clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

size_t ArrayDataBytes(const std::shared_ptr<arrow::ArrayData>& data,
                      std::unordered_set<const arrow::Buffer*>* seen) {
  if (data == nullptr) {
    return 0;
  }
  size_t bytes = 0;
  for (const auto& buffer : data->buffers) {
    if (buffer != nullptr && seen->insert(buffer.get()).second) {
      bytes += static_cast<size_t>(buffer->size());
    }
  }
  for (const auto& child : data->child_data) {
    bytes += ArrayDataBytes(child, seen);
  }
  return bytes;
}

size_t TableBytes(const std::shared_ptr<arrow::Table>& table) {
  if (table == nullptr) {
    return 0;
  }
  std::unordered_set<const arrow::Buffer*> seen;
  size_t bytes = 0;
  for (int col = 0; col < table->num_columns(); ++col) {
    for (const auto& chunk : table->column(col)->chunks()) {
      bytes += ArrayDataBytes(chunk->data(), &seen);
    }
  }
  return bytes;
}

Result<std::shared_ptr<arrow::Table>> LoadVertexPropertyChunkDirect(
    const std::shared_ptr<GraphInfo>& graph_info, const std::string& vertex_type,
    const std::shared_ptr<PropertyGroup>& property_group,
    const std::shared_ptr<VertexInfo>& vertex_info, IdType chunk_id) {
  auto reader_result =
      VertexPropertyArrowChunkReader::Make(graph_info, vertex_type,
                                           property_group);
  GAR_RETURN_NOT_OK(reader_result.status());
  auto reader = reader_result.value();

  const IdType chunk_start_node = chunk_id * vertex_info->GetChunkSize();
  GAR_RETURN_NOT_OK(reader->seek(chunk_start_node));

  auto chunk_result = reader->GetChunk();
  GAR_RETURN_NOT_OK(chunk_result.status());
  return chunk_result.value();
}

}  // namespace

bool ChunkReadKey::operator==(const ChunkReadKey& other) const {
  return kind == other.kind && graph_prefix == other.graph_prefix &&
         vertex_type == other.vertex_type &&
         edge_type == other.edge_type && dst_type == other.dst_type &&
         property_group_prefix == other.property_group_prefix &&
         adj_list_type == other.adj_list_type && file_type == other.file_type &&
         vertex_chunk_id == other.vertex_chunk_id && chunk_id == other.chunk_id;
}

size_t ChunkReadKeyHash::operator()(const ChunkReadKey& key) const {
  size_t seed = 0;
  HashCombine(&seed, static_cast<int>(key.kind));
  HashCombine(&seed, key.graph_prefix);
  HashCombine(&seed, key.vertex_type);
  HashCombine(&seed, key.edge_type);
  HashCombine(&seed, key.dst_type);
  HashCombine(&seed, key.property_group_prefix);
  HashCombine(&seed, static_cast<int>(key.adj_list_type));
  HashCombine(&seed, static_cast<int>(key.file_type));
  HashCombine(&seed, key.vertex_chunk_id);
  HashCombine(&seed, key.chunk_id);
  return seed;
}

struct ChunkReadManager::FeatureCursorState {
  struct TrailEntry {
    ChunkReadKey key;
    TablePtr table;
  };

  struct Task {
    ChunkReadKey key;
    std::shared_ptr<GraphInfo> graph_info;
    std::string vertex_type;
    std::shared_ptr<PropertyGroup> property_group;
    std::shared_ptr<VertexInfo> vertex_info;
    IdType chunk_id = 0;
    std::promise<TableResult> promise;
  };

  explicit FeatureCursorState(const ChunkReadManagerOptions& options)
      : cursor_count_(options.feature_cursor_count),
        trail_capacity_chunks_(options.feature_cursor_trail_capacity_chunks),
        queues_(cursor_count_) {
    workers_.reserve(cursor_count_);
    for (size_t i = 0; i < cursor_count_; ++i) {
      workers_.emplace_back([this, i]() { WorkerLoop(i); });
    }
  }

  ~FeatureCursorState() { Shutdown(); }

  TableResult LoadChunk(const std::shared_ptr<GraphInfo>& graph_info,
                        const std::string& vertex_type,
                        const std::shared_ptr<PropertyGroup>& property_group,
                        const std::shared_ptr<VertexInfo>& vertex_info,
                        const ChunkReadKey& key, IdType chunk_id) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        return Status::Invalid("Feature cursor is shut down");
      }
      auto trail_table = LookupTrailLocked(key);
      if (trail_table != nullptr) {
        trail_hits_.fetch_add(1, std::memory_order_relaxed);
        return trail_table;
      }
      trail_misses_.fetch_add(1, std::memory_order_relaxed);
    }

    auto task = std::make_shared<Task>();
    task->key = key;
    task->graph_info = graph_info;
    task->vertex_type = vertex_type;
    task->property_group = property_group;
    task->vertex_info = vertex_info;
    task->chunk_id = chunk_id;
    auto future = task->promise.get_future();

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        return Status::Invalid("Feature cursor is shut down");
      }
      queues_[CursorIndexFor(key)].push_back(task);
    }
    cv_.notify_all();
    return future.get();
  }

  FeatureCursorStats stats() const {
    FeatureCursorStats stats;
    stats.cursor_count = cursor_count_;
    stats.trail_capacity_chunks = trail_capacity_chunks_;
    stats.requests = requests_.load(std::memory_order_relaxed);
    stats.requests_completed =
        requests_completed_.load(std::memory_order_relaxed);
    stats.requests_failed = requests_failed_.load(std::memory_order_relaxed);
    stats.active_requests_peak =
        active_requests_peak_.load(std::memory_order_relaxed);
    stats.chunks_read = chunks_read_.load(std::memory_order_relaxed);
    stats.chunks_served = chunks_served_.load(std::memory_order_relaxed);
    stats.rows_served = rows_served_.load(std::memory_order_relaxed);
    stats.batches_served = batches_served_.load(std::memory_order_relaxed);
    stats.trail_hits = trail_hits_.load(std::memory_order_relaxed);
    stats.trail_misses = trail_misses_.load(std::memory_order_relaxed);
    stats.trail_evictions = trail_evictions_.load(std::memory_order_relaxed);
    stats.wait_ms_sum = wait_ms_sum_.load(std::memory_order_relaxed);
    stats.wait_ms_max = wait_ms_max_.load(std::memory_order_relaxed);
    stats.service_ms_sum = service_ms_sum_.load(std::memory_order_relaxed);
    stats.service_ms_max = service_ms_max_.load(std::memory_order_relaxed);
    return stats;
  }

  void Shutdown() {
    std::vector<std::shared_ptr<Task>> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        return;
      }
      shutdown_ = true;
      for (auto& queue : queues_) {
        while (!queue.empty()) {
          pending.push_back(std::move(queue.front()));
          queue.pop_front();
        }
      }
    }

    cv_.notify_all();
    for (const auto& task : pending) {
      task->promise.set_value(Status::Invalid("Feature cursor is shut down"));
    }
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  void RegisterRequest() {
    requests_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t active =
        active_requests_current_.fetch_add(1, std::memory_order_relaxed) + 1;
    AtomicMax(&active_requests_peak_, active);
  }

  void CompleteRequest(Clock::time_point start, bool ok) {
    const uint64_t wait_ms = MillisecondsSince(start);
    wait_ms_sum_.fetch_add(wait_ms, std::memory_order_relaxed);
    AtomicMax(&wait_ms_max_, wait_ms);
    active_requests_current_.fetch_sub(1, std::memory_order_relaxed);
    if (ok) {
      requests_completed_.fetch_add(1, std::memory_order_relaxed);
    } else {
      requests_failed_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void RecordBatchServed(size_t rows) {
    rows_served_.fetch_add(rows, std::memory_order_relaxed);
    batches_served_.fetch_add(1, std::memory_order_relaxed);
  }

 private:
  size_t CursorIndexFor(const ChunkReadKey& key) const {
    return ChunkReadKeyHash{}(key) % cursor_count_;
  }

  void WorkerLoop(size_t index) {
    while (true) {
      std::shared_ptr<Task> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() { return shutdown_ || !queues_[index].empty(); });
        if (shutdown_ && queues_[index].empty()) {
          return;
        }
        task = std::move(queues_[index].front());
        queues_[index].pop_front();
      }

      if (IsShutdown()) {
        task->promise.set_value(Status::Invalid("Feature cursor is shut down"));
        continue;
      }

      const auto service_start = Clock::now();
      TableResult result = Status::Invalid("feature trail miss");
      bool served_from_trail = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto trail_table = LookupTrailLocked(task->key);
        if (trail_table != nullptr) {
          trail_hits_.fetch_add(1, std::memory_order_relaxed);
          result = trail_table;
          served_from_trail = true;
        }
      }

      if (served_from_trail) {
        const uint64_t service_ms = MillisecondsSince(service_start);
        service_ms_sum_.fetch_add(service_ms, std::memory_order_relaxed);
        AtomicMax(&service_ms_max_, service_ms);
        task->promise.set_value(result);
        continue;
      }

      result = [&]() -> TableResult {
        try {
          auto chunk_result = LoadVertexPropertyChunkDirect(
              task->graph_info, task->vertex_type, task->property_group,
              task->vertex_info, task->chunk_id);
          if (!chunk_result.has_error()) {
            std::lock_guard<std::mutex> lock(mutex_);
            InsertTrailLocked(task->key, chunk_result.value());
          }
          return chunk_result;
        } catch (const std::exception& e) {
          return Status::UnknownError("Feature cursor chunk read threw: ",
                                      e.what());
        } catch (...) {
          return Status::UnknownError(
              "Feature cursor chunk read threw unknown error");
        }
      }();

      if (!result.has_error()) {
        chunks_read_.fetch_add(1, std::memory_order_relaxed);
        chunks_served_.fetch_add(1, std::memory_order_relaxed);
      } else {
        chunks_read_.fetch_add(1, std::memory_order_relaxed);
      }

      const uint64_t service_ms = MillisecondsSince(service_start);
      service_ms_sum_.fetch_add(service_ms, std::memory_order_relaxed);
      AtomicMax(&service_ms_max_, service_ms);
      task->promise.set_value(result);
    }
  }

  bool IsShutdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_;
  }

  TablePtr LookupTrailLocked(const ChunkReadKey& key) const {
    auto it = trail_index_.find(key);
    if (it == trail_index_.end()) {
      return nullptr;
    }
    return it->second->table;
  }

  void InsertTrailLocked(const ChunkReadKey& key, const TablePtr& table) {
    if (trail_capacity_chunks_ == 0 || table == nullptr) {
      return;
    }
    auto existing = trail_index_.find(key);
    if (existing != trail_index_.end()) {
      existing->second->table = table;
      trail_.splice(trail_.end(), trail_, existing->second);
      return;
    }

    trail_.push_back(TrailEntry{key, table});
    auto inserted = std::prev(trail_.end());
    trail_index_.emplace(key, inserted);
    while (trail_.size() > trail_capacity_chunks_) {
      trail_index_.erase(trail_.front().key);
      trail_.pop_front();
      trail_evictions_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  const size_t cursor_count_;
  const size_t trail_capacity_chunks_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool shutdown_ = false;
  std::vector<std::thread> workers_;
  std::vector<std::deque<std::shared_ptr<Task>>> queues_;
  std::list<TrailEntry> trail_;
  std::unordered_map<ChunkReadKey, std::list<TrailEntry>::iterator,
                     ChunkReadKeyHash>
      trail_index_;

  std::atomic<uint64_t> requests_{0};
  std::atomic<uint64_t> requests_completed_{0};
  std::atomic<uint64_t> requests_failed_{0};
  std::atomic<uint64_t> active_requests_current_{0};
  std::atomic<uint64_t> active_requests_peak_{0};
  std::atomic<uint64_t> chunks_read_{0};
  std::atomic<uint64_t> chunks_served_{0};
  std::atomic<uint64_t> rows_served_{0};
  std::atomic<uint64_t> batches_served_{0};
  std::atomic<uint64_t> trail_hits_{0};
  std::atomic<uint64_t> trail_misses_{0};
  std::atomic<uint64_t> trail_evictions_{0};
  std::atomic<uint64_t> wait_ms_sum_{0};
  std::atomic<uint64_t> wait_ms_max_{0};
  std::atomic<uint64_t> service_ms_sum_{0};
  std::atomic<uint64_t> service_ms_max_{0};
};

ChunkReadManager::ChunkReadManager(ChunkReadManagerOptions options)
    : options_(std::move(options)) {
  if (options_.feature_cursor_count > 0) {
    feature_cursor_ = std::make_unique<FeatureCursorState>(options_);
  }
}

ChunkReadManager::~ChunkReadManager() { Shutdown(); }

bool ChunkReadManager::HasFeatureCursor() const {
  return feature_cursor_ != nullptr;
}

ChunkReadManager::CacheDomain ChunkReadManager::CacheDomainFor(
    const ChunkReadKey& key) const {
  switch (key.kind) {
  case ChunkReadKind::kVertexProperty:
    return CacheDomain::kVertexProperty;
  case ChunkReadKind::kEdgeOffset:
    return CacheDomain::kEdgeOffset;
  case ChunkReadKind::kEdgeAdjList:
    return CacheDomain::kEdgeAdjList;
  }
  return CacheDomain::kVertexProperty;
}

void ChunkReadManager::RecordRamCacheHit(CacheDomain domain) {
  switch (domain) {
  case CacheDomain::kVertexProperty:
    vertex_property_ram_cache_hits_.fetch_add(1, std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeOffset:
    edge_offset_ram_cache_hits_.fetch_add(1, std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeAdjList:
    edge_adj_list_ram_cache_hits_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  return;
}

void ChunkReadManager::RecordRamCacheMiss(CacheDomain domain) {
  switch (domain) {
  case CacheDomain::kVertexProperty:
    vertex_property_ram_cache_misses_.fetch_add(1, std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeOffset:
    edge_offset_ram_cache_misses_.fetch_add(1, std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeAdjList:
    edge_adj_list_ram_cache_misses_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  return;
}

void ChunkReadManager::RecordRamCacheEviction(CacheDomain domain) {
  switch (domain) {
  case CacheDomain::kVertexProperty:
    vertex_property_ram_cache_evictions_.fetch_add(1,
                                                   std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeOffset:
    edge_offset_ram_cache_evictions_.fetch_add(1, std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeAdjList:
    edge_adj_list_ram_cache_evictions_.fetch_add(1,
                                                 std::memory_order_relaxed);
    return;
  }
  return;
}

ChunkReadManager::CacheStore* ChunkReadManager::CacheStoreFor(
    CacheDomain domain) {
  switch (domain) {
  case CacheDomain::kVertexProperty:
    return &vertex_property_ram_cache_;
  case CacheDomain::kEdgeOffset:
    return &edge_offset_ram_cache_;
  case CacheDomain::kEdgeAdjList:
    return &edge_adj_list_ram_cache_;
  }
  return &vertex_property_ram_cache_;
}

const ChunkReadManager::CacheStore* ChunkReadManager::CacheStoreFor(
    CacheDomain domain) const {
  switch (domain) {
  case CacheDomain::kVertexProperty:
    return &vertex_property_ram_cache_;
  case CacheDomain::kEdgeOffset:
    return &edge_offset_ram_cache_;
  case CacheDomain::kEdgeAdjList:
    return &edge_adj_list_ram_cache_;
  }
  return &vertex_property_ram_cache_;
}

size_t ChunkReadManager::RamBudgetBytesFor(CacheDomain domain) const {
  switch (domain) {
  case CacheDomain::kVertexProperty:
    return options_.ram_budget_bytes;
  case CacheDomain::kEdgeOffset:
    return options_.edge_offset_ram_budget_bytes;
  case CacheDomain::kEdgeAdjList:
    return options_.edge_adj_list_ram_budget_bytes;
  }
  return options_.ram_budget_bytes;
}

void ChunkReadManager::RegisterFeatureRequest() {
  if (feature_cursor_ != nullptr) {
    feature_cursor_->RegisterRequest();
  }
}

void ChunkReadManager::CompleteFeatureRequest(Clock::time_point start, bool ok) {
  if (feature_cursor_ != nullptr) {
    feature_cursor_->CompleteRequest(start, ok);
  }
}

void ChunkReadManager::RecordFeatureBatchServed(size_t rows) {
  if (feature_cursor_ != nullptr) {
    feature_cursor_->RecordBatchServed(rows);
  }
}

ChunkReadManager::TableResult ChunkReadManager::GetOrLoad(
    const ChunkReadKey& key, const TableLoader& loader) {
  requests_.fetch_add(1, std::memory_order_relaxed);
  const auto domain = CacheDomainFor(key);
  const size_t ram_budget_bytes = RamBudgetBytesFor(domain);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto cached = LookupRamCacheLocked(key);
    if (cached != nullptr) {
      RecordRamCacheHit(domain);
      TableResult result = cached;
      RecordResult(result);
      return result;
    }
    if (ram_budget_bytes > 0) {
      RecordRamCacheMiss(domain);
    }
  }

  if (!options_.enable_singleflight) {
    leaders_.fetch_add(1, std::memory_order_relaxed);
    TableResult result = loader();
    if (!result.has_error()) {
      std::lock_guard<std::mutex> lock(mutex_);
      InsertRamCacheLocked(key, result.value());
    }
    RecordResult(result);
    return result;
  }

  std::shared_ptr<std::promise<TableResult>> promise;
  std::shared_future<TableResult> future;
  bool is_leader = false;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = in_flight_.find(key);
    if (it != in_flight_.end()) {
      future = it->second;
      waiters_.fetch_add(1, std::memory_order_relaxed);
    } else {
      promise = std::make_shared<std::promise<TableResult>>();
      future = promise->get_future().share();
      in_flight_.emplace(key, future);
      is_leader = true;
      leaders_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  if (!is_leader) {
    TableResult result = future.get();
    RecordResult(result);
    return result;
  }

  TableResult result = [&]() -> TableResult {
    try {
      return loader();
    } catch (const std::exception& e) {
      return Status::UnknownError("Chunk loader threw exception: ", e.what());
    } catch (...) {
      return Status::UnknownError("Chunk loader threw unknown exception");
    }
  }();

  if (!result.has_error()) {
    std::lock_guard<std::mutex> lock(mutex_);
    InsertRamCacheLocked(key, result.value());
  }

  promise->set_value(result);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    in_flight_.erase(key);
  }

  RecordResult(result);
  return result;
}

ChunkReadManager::TableResult ChunkReadManager::GetVertexPropertyChunk(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::string& vertex_type,
    const std::shared_ptr<PropertyGroup>& property_group, IdType chunk_id) {
  if (graph_info == nullptr) {
    return Status::Invalid("GraphInfo cannot be null");
  }
  if (property_group == nullptr) {
    return Status::Invalid("PropertyGroup cannot be null");
  }
  auto vertex_info = graph_info->GetVertexInfo(vertex_type);
  if (!vertex_info) {
    return Status::Invalid("Vertex type '", vertex_type, "' not found");
  }

  ChunkReadKey key;
  key.kind = ChunkReadKind::kVertexProperty;
  key.graph_prefix = graph_info->GetPrefix();
  key.vertex_type = vertex_type;
  key.property_group_prefix = property_group->GetPrefix();
  key.file_type = property_group->GetFileType();
  key.chunk_id = chunk_id;

  if (feature_cursor_ != nullptr) {
    return GetOrLoad(key, [this, graph_info, vertex_type, property_group,
                           vertex_info, key, chunk_id]() -> TableResult {
      return feature_cursor_->LoadChunk(graph_info, vertex_type, property_group,
                                        vertex_info, key, chunk_id);
    });
  }

  return GetOrLoad(key, [graph_info, vertex_type, property_group, vertex_info,
                         chunk_id]() -> TableResult {
    return LoadVertexPropertyChunkDirect(graph_info, vertex_type, property_group,
                                         vertex_info, chunk_id);
  });
}

ChunkReadManager::TableResult ChunkReadManager::GetEdgeOffsetChunk(
    const std::shared_ptr<GraphInfo>& graph_info, const std::string& src_type,
    const std::string& edge_type, const std::string& dst_type,
    AdjListType adj_list_type, IdType vertex_chunk_id) {
  if (graph_info == nullptr) {
    return Status::Invalid("GraphInfo cannot be null");
  }
  auto edge_info = graph_info->GetEdgeInfo(src_type, edge_type, dst_type);
  if (!edge_info) {
    return Status::Invalid("Edge type '", edge_type, "' not found");
  }
  if (!edge_info->HasAdjacentListType(adj_list_type)) {
    return Status::Invalid("Adjacent list type not available for edge type '",
                           edge_type, "'");
  }

  ChunkReadKey key;
  key.kind = ChunkReadKind::kEdgeOffset;
  key.graph_prefix = graph_info->GetPrefix();
  key.vertex_type = src_type;
  key.edge_type = edge_type;
  key.dst_type = dst_type;
  key.adj_list_type = adj_list_type;
  key.file_type = edge_info->GetAdjacentList(adj_list_type)->GetFileType();
  key.vertex_chunk_id = vertex_chunk_id;

  return GetOrLoad(key, [graph_info, edge_info, adj_list_type,
                         vertex_chunk_id]() -> TableResult {
    const std::string& prefix = graph_info->GetPrefix();
    std::string normalized_prefix;
    GAR_ASSIGN_OR_RAISE(auto fs,
                        FileSystemFromUriOrPath(prefix, &normalized_prefix));
    GAR_ASSIGN_OR_RAISE(
        auto chunk_file_path,
        edge_info->GetAdjListOffsetFilePath(vertex_chunk_id, adj_list_type));
    auto file_type = edge_info->GetAdjacentList(adj_list_type)->GetFileType();
    GAR_ASSIGN_OR_RAISE(auto table,
                        fs->ReadFileToTable(normalized_prefix + chunk_file_path,
                                            file_type));
    if (table->num_columns() == 0) {
      return Status::Invalid("Offset file for edge type '",
                             edge_info->GetEdgeType(), "' has no columns");
    }
    return table;
  });
}

ChunkReadManager::TableResult ChunkReadManager::GetEdgeAdjListChunk(
    const std::shared_ptr<GraphInfo>& graph_info, const std::string& src_type,
    const std::string& edge_type, const std::string& dst_type,
    AdjListType adj_list_type, IdType vertex_chunk_id, IdType chunk_id) {
  if (graph_info == nullptr) {
    return Status::Invalid("GraphInfo cannot be null");
  }
  auto edge_info = graph_info->GetEdgeInfo(src_type, edge_type, dst_type);
  if (!edge_info) {
    return Status::Invalid("Edge type '", edge_type, "' not found");
  }
  if (!edge_info->HasAdjacentListType(adj_list_type)) {
    return Status::Invalid("Adjacent list type not available for edge type '",
                           edge_type, "'");
  }

  ChunkReadKey key;
  key.kind = ChunkReadKind::kEdgeAdjList;
  key.graph_prefix = graph_info->GetPrefix();
  key.vertex_type = src_type;
  key.edge_type = edge_type;
  key.dst_type = dst_type;
  key.adj_list_type = adj_list_type;
  key.file_type = edge_info->GetAdjacentList(adj_list_type)->GetFileType();
  key.vertex_chunk_id = vertex_chunk_id;
  key.chunk_id = chunk_id;

  return GetOrLoad(key, [edge_info, adj_list_type, graph_info, vertex_chunk_id,
                         chunk_id]() -> TableResult {
    AdjListArrowChunkReader reader(edge_info, adj_list_type,
                                   graph_info->GetPrefix());
    GAR_RETURN_NOT_OK(reader.seek_chunk_index(vertex_chunk_id, chunk_id));
    auto chunk_result = reader.GetChunk();
    GAR_RETURN_NOT_OK(chunk_result.status());
    return chunk_result.value();
  });
}

ChunkReadStats ChunkReadManager::stats() const {
  ChunkReadStats stats;
  stats.requests = requests_.load(std::memory_order_relaxed);
  stats.leaders = leaders_.load(std::memory_order_relaxed);
  stats.waiters = waiters_.load(std::memory_order_relaxed);
  stats.completed = completed_.load(std::memory_order_relaxed);
  stats.failed = failed_.load(std::memory_order_relaxed);
  stats.vertex_property_ram_cache_hits =
      vertex_property_ram_cache_hits_.load(std::memory_order_relaxed);
  stats.vertex_property_ram_cache_misses =
      vertex_property_ram_cache_misses_.load(std::memory_order_relaxed);
  stats.vertex_property_ram_cache_evictions =
      vertex_property_ram_cache_evictions_.load(std::memory_order_relaxed);
  stats.edge_offset_ram_cache_hits =
      edge_offset_ram_cache_hits_.load(std::memory_order_relaxed);
  stats.edge_offset_ram_cache_misses =
      edge_offset_ram_cache_misses_.load(std::memory_order_relaxed);
  stats.edge_offset_ram_cache_evictions =
      edge_offset_ram_cache_evictions_.load(std::memory_order_relaxed);
  stats.edge_adj_list_ram_cache_hits =
      edge_adj_list_ram_cache_hits_.load(std::memory_order_relaxed);
  stats.edge_adj_list_ram_cache_misses =
      edge_adj_list_ram_cache_misses_.load(std::memory_order_relaxed);
  stats.edge_adj_list_ram_cache_evictions =
      edge_adj_list_ram_cache_evictions_.load(std::memory_order_relaxed);
  stats.ram_cache_hits = stats.vertex_property_ram_cache_hits +
                         stats.edge_offset_ram_cache_hits +
                         stats.edge_adj_list_ram_cache_hits;
  stats.ram_cache_misses = stats.vertex_property_ram_cache_misses +
                           stats.edge_offset_ram_cache_misses +
                           stats.edge_adj_list_ram_cache_misses;
  stats.ram_cache_evictions = stats.vertex_property_ram_cache_evictions +
                              stats.edge_offset_ram_cache_evictions +
                              stats.edge_adj_list_ram_cache_evictions;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stats.vertex_property_ram_cache_bytes = vertex_property_ram_cache_.bytes;
    stats.edge_offset_ram_cache_bytes = edge_offset_ram_cache_.bytes;
    stats.edge_adj_list_ram_cache_bytes = edge_adj_list_ram_cache_.bytes;
    stats.ram_cache_bytes = stats.vertex_property_ram_cache_bytes +
                            stats.edge_offset_ram_cache_bytes +
                            stats.edge_adj_list_ram_cache_bytes;
  }
  return stats;
}

FeatureCursorStats ChunkReadManager::feature_cursor_stats() const {
  if (feature_cursor_ == nullptr) {
    return {};
  }
  return feature_cursor_->stats();
}

void ChunkReadManager::Shutdown() {
  if (feature_cursor_ != nullptr) {
    feature_cursor_->Shutdown();
  }
}

void ChunkReadManager::RecordResult(const TableResult& result) {
  if (result.has_error()) {
    failed_.fetch_add(1, std::memory_order_relaxed);
  } else {
    completed_.fetch_add(1, std::memory_order_relaxed);
  }
}

ChunkReadManager::TablePtr ChunkReadManager::LookupRamCacheLocked(
    const ChunkReadKey& key) {
  auto* store = CacheStoreFor(CacheDomainFor(key));
  auto it = store->entries.find(key);
  if (it == store->entries.end()) {
    return nullptr;
  }
  store->lru.splice(store->lru.end(), store->lru, it->second.lru_it);
  return it->second.table;
}

void ChunkReadManager::InsertRamCacheLocked(const ChunkReadKey& key,
                                            const TablePtr& table) {
  const auto domain = CacheDomainFor(key);
  const size_t ram_budget_bytes = RamBudgetBytesFor(domain);
  if (ram_budget_bytes == 0 || table == nullptr) {
    return;
  }

  const size_t bytes = TableBytes(table);
  if (bytes > ram_budget_bytes) {
    return;
  }

  auto* store = CacheStoreFor(domain);
  auto existing = store->entries.find(key);
  if (existing != store->entries.end()) {
    store->bytes -= existing->second.bytes;
    store->lru.erase(existing->second.lru_it);
    store->entries.erase(existing);
  }

  EvictRamCacheLocked(domain, bytes);
  store->lru.push_back(key);
  auto lru_it = std::prev(store->lru.end());
  store->entries.emplace(key, CacheEntry{table, bytes, lru_it});
  store->bytes += bytes;
}

void ChunkReadManager::EvictRamCacheLocked(CacheDomain domain,
                                           size_t bytes_needed) {
  auto* store = CacheStoreFor(domain);
  const size_t ram_budget_bytes = RamBudgetBytesFor(domain);
  while (store->bytes + bytes_needed > ram_budget_bytes &&
         !store->lru.empty()) {
    const auto& victim_key = store->lru.front();
    auto victim = store->entries.find(victim_key);
    if (victim != store->entries.end()) {
      store->bytes -= victim->second.bytes;
      store->entries.erase(victim);
      RecordRamCacheEviction(domain);
    }
    store->lru.pop_front();
  }
}

}  // namespace graphar::ml
