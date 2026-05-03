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
#include <limits>
#include <map>
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

Result<std::shared_ptr<arrow::Table>> LoadEdgeOffsetChunkDirect(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::shared_ptr<EdgeInfo>& edge_info, AdjListType adj_list_type,
    IdType vertex_chunk_id) {
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
    return Status::Invalid("Offset file for edge type '", edge_info->GetEdgeType(),
                           "' has no columns");
  }
  return table;
}

Result<std::shared_ptr<arrow::Table>> LoadEdgeAdjListChunkDirect(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::shared_ptr<EdgeInfo>& edge_info, AdjListType adj_list_type,
    IdType vertex_chunk_id, IdType chunk_id) {
  AdjListArrowChunkReader reader(edge_info, adj_list_type, graph_info->GetPrefix());
  GAR_RETURN_NOT_OK(reader.seek_chunk_index(vertex_chunk_id, chunk_id));
  auto chunk_result = reader.GetChunk();
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

ChunkReadManager::ChunkReadManager(ChunkReadManagerOptions options)
    : options_(std::move(options)) {
  if (options_.feature_cursor_count > 0) {
    feature_cursor_ = std::make_unique<OrderedChunkCursor>(
        options_.feature_cursor_count,
        options_.feature_cursor_trail_capacity_chunks, false);
  }
  if (options_.edge_cursor_count > 0) {
    edge_offset_cursor_ = std::make_unique<OrderedChunkCursor>(
        options_.edge_cursor_count,
        options_.edge_cursor_trail_capacity_chunks, true);
    edge_adj_list_cursor_ = std::make_unique<OrderedChunkCursor>(
        options_.edge_cursor_count,
        options_.edge_cursor_trail_capacity_chunks, true);
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

void ChunkReadManager::RecordRequest(CacheDomain domain) {
  requests_.fetch_add(1, std::memory_order_relaxed);
  switch (domain) {
  case CacheDomain::kVertexProperty:
    vertex_property_requests_.fetch_add(1, std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeOffset:
    edge_offset_requests_.fetch_add(1, std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeAdjList:
    edge_adj_list_requests_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  return;
}

void ChunkReadManager::RecordLeader(CacheDomain domain) {
  leaders_.fetch_add(1, std::memory_order_relaxed);
  switch (domain) {
  case CacheDomain::kVertexProperty:
    vertex_property_leaders_.fetch_add(1, std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeOffset:
    edge_offset_leaders_.fetch_add(1, std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeAdjList:
    edge_adj_list_leaders_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  return;
}

void ChunkReadManager::RecordWaiter(CacheDomain domain) {
  waiters_.fetch_add(1, std::memory_order_relaxed);
  switch (domain) {
  case CacheDomain::kVertexProperty:
    vertex_property_waiters_.fetch_add(1, std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeOffset:
    edge_offset_waiters_.fetch_add(1, std::memory_order_relaxed);
    return;
  case CacheDomain::kEdgeAdjList:
    edge_adj_list_waiters_.fetch_add(1, std::memory_order_relaxed);
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
  const auto domain = CacheDomainFor(key);
  RecordRequest(domain);
  const size_t ram_budget_bytes = RamBudgetBytesFor(domain);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto cached = LookupRamCacheLocked(key);
    if (cached != nullptr) {
      RecordRamCacheHit(domain);
      TableResult result = cached;
      RecordResult(domain, result);
      return result;
    }
    if (ram_budget_bytes > 0) {
      RecordRamCacheMiss(domain);
    }
  }

  if (!options_.enable_singleflight) {
    RecordLeader(domain);
    TableResult result = loader();
    if (!result.has_error()) {
      std::lock_guard<std::mutex> lock(mutex_);
      InsertRamCacheLocked(key, result.value());
    }
    RecordResult(domain, result);
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
      RecordWaiter(domain);
    } else {
      promise = std::make_shared<std::promise<TableResult>>();
      future = promise->get_future().share();
      in_flight_.emplace(key, future);
      is_leader = true;
      RecordLeader(domain);
    }
  }

  if (!is_leader) {
    TableResult result = future.get();
    RecordResult(domain, result);
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

  RecordResult(domain, result);
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
      return feature_cursor_->LoadChunk(
          key, OrderedChunkCursor::OrderKey{0, 0, chunk_id},
          [graph_info, vertex_type, property_group, vertex_info,
           chunk_id]() -> TableResult {
            return LoadVertexPropertyChunkDirect(graph_info, vertex_type,
                                                 property_group, vertex_info,
                                                 chunk_id);
          });
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

  if (edge_offset_cursor_ != nullptr) {
    return GetOrLoad(
        key, [this, key, graph_info, edge_info, adj_list_type,
              vertex_chunk_id]() -> TableResult {
          return edge_offset_cursor_->LoadChunk(
              key, OrderedChunkCursor::OrderKey{vertex_chunk_id, 0, 0},
              [graph_info, edge_info, adj_list_type,
               vertex_chunk_id]() -> TableResult {
                return LoadEdgeOffsetChunkDirect(graph_info, edge_info,
                                                 adj_list_type,
                                                 vertex_chunk_id);
              });
        });
  }

  return GetOrLoad(key, [graph_info, edge_info, adj_list_type,
                         vertex_chunk_id]() -> TableResult {
    return LoadEdgeOffsetChunkDirect(graph_info, edge_info, adj_list_type,
                                     vertex_chunk_id);
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

  if (edge_adj_list_cursor_ != nullptr) {
    return GetOrLoad(
        key, [this, key, graph_info, edge_info, adj_list_type,
              vertex_chunk_id, chunk_id]() -> TableResult {
          return edge_adj_list_cursor_->LoadChunk(
              key, OrderedChunkCursor::OrderKey{vertex_chunk_id, 0, chunk_id},
              [graph_info, edge_info, adj_list_type, vertex_chunk_id,
               chunk_id]() -> TableResult {
                return LoadEdgeAdjListChunkDirect(graph_info, edge_info,
                                                  adj_list_type,
                                                  vertex_chunk_id, chunk_id);
              });
        });
  }

  return GetOrLoad(key, [graph_info, edge_info, adj_list_type, vertex_chunk_id,
                         chunk_id]() -> TableResult {
    return LoadEdgeAdjListChunkDirect(graph_info, edge_info, adj_list_type,
                                      vertex_chunk_id, chunk_id);
  });
}

ChunkReadStats ChunkReadManager::stats() const {
  ChunkReadStats stats;
  stats.requests = requests_.load(std::memory_order_relaxed);
  stats.leaders = leaders_.load(std::memory_order_relaxed);
  stats.waiters = waiters_.load(std::memory_order_relaxed);
  stats.completed = completed_.load(std::memory_order_relaxed);
  stats.failed = failed_.load(std::memory_order_relaxed);
  stats.vertex_property_requests =
      vertex_property_requests_.load(std::memory_order_relaxed);
  stats.vertex_property_leaders =
      vertex_property_leaders_.load(std::memory_order_relaxed);
  stats.vertex_property_waiters =
      vertex_property_waiters_.load(std::memory_order_relaxed);
  stats.vertex_property_completed =
      vertex_property_completed_.load(std::memory_order_relaxed);
  stats.vertex_property_failed =
      vertex_property_failed_.load(std::memory_order_relaxed);
  stats.vertex_property_ram_cache_hits =
      vertex_property_ram_cache_hits_.load(std::memory_order_relaxed);
  stats.vertex_property_ram_cache_misses =
      vertex_property_ram_cache_misses_.load(std::memory_order_relaxed);
  stats.vertex_property_ram_cache_evictions =
      vertex_property_ram_cache_evictions_.load(std::memory_order_relaxed);
  stats.edge_offset_requests =
      edge_offset_requests_.load(std::memory_order_relaxed);
  stats.edge_offset_leaders =
      edge_offset_leaders_.load(std::memory_order_relaxed);
  stats.edge_offset_waiters =
      edge_offset_waiters_.load(std::memory_order_relaxed);
  stats.edge_offset_completed =
      edge_offset_completed_.load(std::memory_order_relaxed);
  stats.edge_offset_failed =
      edge_offset_failed_.load(std::memory_order_relaxed);
  stats.edge_offset_ram_cache_hits =
      edge_offset_ram_cache_hits_.load(std::memory_order_relaxed);
  stats.edge_offset_ram_cache_misses =
      edge_offset_ram_cache_misses_.load(std::memory_order_relaxed);
  stats.edge_offset_ram_cache_evictions =
      edge_offset_ram_cache_evictions_.load(std::memory_order_relaxed);
  stats.edge_adj_list_requests =
      edge_adj_list_requests_.load(std::memory_order_relaxed);
  stats.edge_adj_list_leaders =
      edge_adj_list_leaders_.load(std::memory_order_relaxed);
  stats.edge_adj_list_waiters =
      edge_adj_list_waiters_.load(std::memory_order_relaxed);
  stats.edge_adj_list_completed =
      edge_adj_list_completed_.load(std::memory_order_relaxed);
  stats.edge_adj_list_failed =
      edge_adj_list_failed_.load(std::memory_order_relaxed);
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

FeatureCursorStats ChunkReadManager::edge_offset_cursor_stats() const {
  if (edge_offset_cursor_ == nullptr) {
    return {};
  }
  return edge_offset_cursor_->stats();
}

FeatureCursorStats ChunkReadManager::edge_adj_list_cursor_stats() const {
  if (edge_adj_list_cursor_ == nullptr) {
    return {};
  }
  return edge_adj_list_cursor_->stats();
}

void ChunkReadManager::Shutdown() {
  if (feature_cursor_ != nullptr) {
    feature_cursor_->Shutdown();
  }
  if (edge_offset_cursor_ != nullptr) {
    edge_offset_cursor_->Shutdown();
  }
  if (edge_adj_list_cursor_ != nullptr) {
    edge_adj_list_cursor_->Shutdown();
  }
}

void ChunkReadManager::RecordResult(CacheDomain domain,
                                    const TableResult& result) {
  if (result.has_error()) {
    failed_.fetch_add(1, std::memory_order_relaxed);
    switch (domain) {
    case CacheDomain::kVertexProperty:
      vertex_property_failed_.fetch_add(1, std::memory_order_relaxed);
      return;
    case CacheDomain::kEdgeOffset:
      edge_offset_failed_.fetch_add(1, std::memory_order_relaxed);
      return;
    case CacheDomain::kEdgeAdjList:
      edge_adj_list_failed_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  } else {
    completed_.fetch_add(1, std::memory_order_relaxed);
    switch (domain) {
    case CacheDomain::kVertexProperty:
      vertex_property_completed_.fetch_add(1, std::memory_order_relaxed);
      return;
    case CacheDomain::kEdgeOffset:
      edge_offset_completed_.fetch_add(1, std::memory_order_relaxed);
      return;
    case CacheDomain::kEdgeAdjList:
      edge_adj_list_completed_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
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
