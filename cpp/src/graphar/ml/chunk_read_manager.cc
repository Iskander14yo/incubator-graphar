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
#include <utility>

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

  if (!options_.enable_singleflight) {
    leaders_.fetch_add(1, std::memory_order_relaxed);
    TableResult result = loader();
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
  return stats;
}

void ChunkReadManager::RecordResult(const TableResult& result) {
  if (result.has_error()) {
    failed_.fetch_add(1, std::memory_order_relaxed);
  } else {
    completed_.fetch_add(1, std::memory_order_relaxed);
  }
}

}  // namespace graphar::ml
