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

#include <exception>
#include <iterator>
#include <unordered_set>
#include <utility>

#include "arrow/api.h"
#include "graphar/arrow/chunk_reader.h"
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

}  // namespace

bool ChunkReadKey::operator==(const ChunkReadKey& other) const {
  return kind == other.kind && graph_prefix == other.graph_prefix &&
         vertex_type == other.vertex_type &&
         property_group_prefix == other.property_group_prefix &&
         file_type == other.file_type && chunk_id == other.chunk_id;
}

size_t ChunkReadKeyHash::operator()(const ChunkReadKey& key) const {
  size_t seed = 0;
  HashCombine(&seed, static_cast<int>(key.kind));
  HashCombine(&seed, key.graph_prefix);
  HashCombine(&seed, key.vertex_type);
  HashCombine(&seed, key.property_group_prefix);
  HashCombine(&seed, static_cast<int>(key.file_type));
  HashCombine(&seed, key.chunk_id);
  return seed;
}

ChunkReadManager::ChunkReadManager(ChunkReadManagerOptions options)
    : options_(std::move(options)) {}

ChunkReadManager::TableResult ChunkReadManager::GetOrLoad(
    const ChunkReadKey& key, const TableLoader& loader) {
  requests_.fetch_add(1, std::memory_order_relaxed);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto cached = LookupRamCacheLocked(key);
    if (cached != nullptr) {
      ram_cache_hits_.fetch_add(1, std::memory_order_relaxed);
      TableResult result = cached;
      RecordResult(result);
      return result;
    }
    if (options_.ram_budget_bytes > 0) {
      ram_cache_misses_.fetch_add(1, std::memory_order_relaxed);
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

  return GetOrLoad(key, [graph_info, vertex_type, property_group, vertex_info,
                         chunk_id]() -> TableResult {
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
  });
}

ChunkReadStats ChunkReadManager::stats() const {
  ChunkReadStats stats;
  stats.requests = requests_.load(std::memory_order_relaxed);
  stats.leaders = leaders_.load(std::memory_order_relaxed);
  stats.waiters = waiters_.load(std::memory_order_relaxed);
  stats.completed = completed_.load(std::memory_order_relaxed);
  stats.failed = failed_.load(std::memory_order_relaxed);
  stats.ram_cache_hits = ram_cache_hits_.load(std::memory_order_relaxed);
  stats.ram_cache_misses = ram_cache_misses_.load(std::memory_order_relaxed);
  stats.ram_cache_evictions =
      ram_cache_evictions_.load(std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stats.ram_cache_bytes = ram_cache_bytes_;
  }
  return stats;
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
  auto it = ram_cache_.find(key);
  if (it == ram_cache_.end()) {
    return nullptr;
  }
  ram_cache_lru_.splice(ram_cache_lru_.end(), ram_cache_lru_,
                        it->second.lru_it);
  return it->second.table;
}

void ChunkReadManager::InsertRamCacheLocked(const ChunkReadKey& key,
                                            const TablePtr& table) {
  if (options_.ram_budget_bytes == 0 || table == nullptr) {
    return;
  }

  const size_t bytes = TableBytes(table);
  if (bytes > options_.ram_budget_bytes) {
    return;
  }

  auto existing = ram_cache_.find(key);
  if (existing != ram_cache_.end()) {
    ram_cache_bytes_ -= existing->second.bytes;
    ram_cache_lru_.erase(existing->second.lru_it);
    ram_cache_.erase(existing);
  }

  EvictRamCacheLocked(bytes);
  ram_cache_lru_.push_back(key);
  auto lru_it = std::prev(ram_cache_lru_.end());
  ram_cache_.emplace(key, CacheEntry{table, bytes, lru_it});
  ram_cache_bytes_ += bytes;
}

void ChunkReadManager::EvictRamCacheLocked(size_t bytes_needed) {
  while (ram_cache_bytes_ + bytes_needed > options_.ram_budget_bytes &&
         !ram_cache_lru_.empty()) {
    const auto& victim_key = ram_cache_lru_.front();
    auto victim = ram_cache_.find(victim_key);
    if (victim != ram_cache_.end()) {
      ram_cache_bytes_ -= victim->second.bytes;
      ram_cache_.erase(victim);
      ram_cache_evictions_.fetch_add(1, std::memory_order_relaxed);
    }
    ram_cache_lru_.pop_front();
  }
}

}  // namespace graphar::ml
