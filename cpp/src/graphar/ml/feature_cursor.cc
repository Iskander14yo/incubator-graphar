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

#include "graphar/ml/feature_cursor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <list>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "arrow/api.h"
#include "arrow/compute/api.h"
#include "graphar/graph_info.h"
#include "graphar/ml/chunk_read_manager.h"
#include "graphar/reader_util.h"
#include "graphar/status.h"
#include "graphar/types.h"

namespace graphar::ml {
namespace {

using Clock = std::chrono::steady_clock;

template <typename T>
void HashCombine(size_t* seed, const T& value) {
  std::hash<T> hasher;
  *seed ^= hasher(value) + 0x9e3779b97f4a7c15ULL + (*seed << 6) + (*seed >> 2);
}

void AtomicMax(std::atomic<uint64_t>* target, uint64_t value) {
  uint64_t current = target->load(std::memory_order_relaxed);
  while (current < value && !target->compare_exchange_weak(
                                current, value, std::memory_order_relaxed)) {}
}

uint64_t MillisecondsSince(Clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                            start)
          .count());
}

struct FeatureChunkKey {
  std::string vertex_type;
  std::shared_ptr<PropertyGroup> property_group;
  std::string property_group_prefix;
  FileType file_type = FileType::PARQUET;
  IdType chunk_id = 0;

  bool operator==(const FeatureChunkKey& other) const {
    return vertex_type == other.vertex_type &&
           property_group == other.property_group &&
           property_group_prefix == other.property_group_prefix &&
           file_type == other.file_type && chunk_id == other.chunk_id;
  }
};

struct FeatureChunkKeyHash {
  size_t operator()(const FeatureChunkKey& key) const {
    size_t seed = 0;
    HashCombine(&seed, key.vertex_type);
    HashCombine(&seed, key.property_group.get());
    HashCombine(&seed, key.property_group_prefix);
    HashCombine(&seed, static_cast<int>(key.file_type));
    HashCombine(&seed, key.chunk_id);
    return seed;
  }
};

struct FeatureChunkRows {
  std::vector<IdType> row_offsets;
  std::vector<int64_t> output_positions;
};

struct FeatureGroupPlan {
  std::shared_ptr<PropertyGroup> property_group;
  std::vector<std::string> properties;
};

struct FeatureRequestState {
  using TablePtr = std::shared_ptr<arrow::Table>;
  using TableResult = Result<TablePtr>;

  uint64_t request_id = 0;
  std::shared_ptr<VertexInfo> vertex_info;
  std::vector<std::string> requested_properties;
  size_t row_count = 0;
  Clock::time_point submit_time;
  std::promise<TableResult> promise;

  mutable std::mutex mutex;
  bool completed = false;
  size_t pending_chunks = 0;
  std::unordered_map<FeatureChunkKey, FeatureChunkRows, FeatureChunkKeyHash>
      rows_by_key;
  std::unordered_map<FeatureChunkKey, std::vector<std::string>,
                     FeatureChunkKeyHash>
      properties_by_key;
  std::unordered_map<std::string, std::vector<std::shared_ptr<arrow::Array>>>
      arrays_by_property;
  std::unordered_map<std::string, std::vector<int64_t>> positions_by_property;
};

std::shared_ptr<arrow::Array> MakeInt64Array(
    const std::vector<int64_t>& values) {
  arrow::Int64Builder builder;
  auto status = builder.AppendValues(values);
  if (!status.ok()) {
    return nullptr;
  }
  std::shared_ptr<arrow::Array> array;
  status = builder.Finish(&array);
  if (!status.ok()) {
    return nullptr;
  }
  return array;
}

FeatureRequestState::TableResult BuildFinalTableLocked(
    const FeatureRequestState& request) {
  std::vector<std::shared_ptr<arrow::Field>> schema_fields;
  std::vector<std::shared_ptr<arrow::Array>> result_arrays;
  schema_fields.reserve(request.requested_properties.size());
  result_arrays.reserve(request.requested_properties.size());

  for (const auto& prop : request.requested_properties) {
    auto arrays_it = request.arrays_by_property.find(prop);
    auto positions_it = request.positions_by_property.find(prop);
    if (arrays_it == request.arrays_by_property.end() ||
        positions_it == request.positions_by_property.end()) {
      return Status::Invalid("No feature data collected for property '", prop,
                             "'");
    }
    if (positions_it->second.size() != request.row_count) {
      return Status::Invalid("Collected ", positions_it->second.size(),
                             " rows for property '", prop, "', expected ",
                             request.row_count);
    }

    auto concat_result = arrow::Concatenate(arrays_it->second);
    if (!concat_result.ok()) {
      return Status::ArrowError(concat_result.status().ToString());
    }

    std::vector<int64_t> inverse(request.row_count);
    for (size_t i = 0; i < positions_it->second.size(); ++i) {
      const int64_t output_pos = positions_it->second[i];
      if (output_pos < 0 ||
          output_pos >= static_cast<int64_t>(request.row_count)) {
        return Status::Invalid("Feature output position out of range");
      }
      inverse[static_cast<size_t>(output_pos)] = static_cast<int64_t>(i);
    }
    auto perm_array = MakeInt64Array(inverse);
    if (perm_array == nullptr) {
      return Status::ArrowError("Failed to build feature permutation array");
    }

    auto reorder_result =
        arrow::compute::Take(concat_result.ValueOrDie(), perm_array);
    if (!reorder_result.ok()) {
      return Status::ArrowError(reorder_result.status().ToString());
    }

    auto prop_type_result = request.vertex_info->GetPropertyType(prop);
    GAR_RETURN_NOT_OK(prop_type_result.status());
    auto arrow_type =
        DataType::DataTypeToArrowDataType(prop_type_result.value());
    schema_fields.push_back(arrow::field(prop, arrow_type));
    result_arrays.push_back(reorder_result.ValueOrDie().make_array());
  }

  return arrow::Table::Make(arrow::schema(schema_fields), result_arrays,
                            request.row_count);
}

}  // namespace

FeatureRequestHandle::FeatureRequestHandle(
    std::shared_future<TableResult> future)
    : future_(std::move(future)) {}

FeatureRequestHandle::TableResult FeatureRequestHandle::Wait() const {
  return future_.get();
}

struct FeatureScanCoordinator::Impl {
  using RequestPtr = std::shared_ptr<FeatureRequestState>;
  using TablePtr = std::shared_ptr<arrow::Table>;
  using TableResult = Result<TablePtr>;

  Impl(std::shared_ptr<GraphInfo> graph_info,
       std::shared_ptr<ChunkReadManager> chunk_manager,
       FeatureCursorOptions options)
      : graph_info_(std::move(graph_info)),
        chunk_manager_(std::move(chunk_manager)),
        options_(options) {
    options_.cursor_count = std::max<size_t>(1, options_.cursor_count);
  }

  ~Impl() { Shutdown(); }

  FeatureScanCoordinator::HandleResult Submit(
      const std::string& vertex_type, const std::vector<IdType>& node_ids,
      const std::vector<std::string>& properties) {
    if (graph_info_ == nullptr) {
      return Status::Invalid("GraphInfo cannot be null");
    }
    if (chunk_manager_ == nullptr) {
      return Status::Invalid("ChunkReadManager cannot be null");
    }

    auto request = std::make_shared<FeatureRequestState>();
    request->submit_time = Clock::now();
    auto future = request->promise.get_future().share();
    auto handle = std::make_shared<FeatureRequestHandle>(future);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        return Status::Invalid("FeatureScanCoordinator is shut down");
      }
    }

    if (node_ids.empty()) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) {
          return Status::Invalid("FeatureScanCoordinator is shut down");
        }
        RegisterRequestLocked();
        request->request_id = next_request_id_.fetch_add(1);
      }
      CompleteRequest(
          request,
          arrow::Table::Make(arrow::schema({}),
                             std::vector<std::shared_ptr<arrow::Array>>{}, 0));
      return handle;
    }
    if (properties.empty()) {
      return Status::Invalid("Properties list cannot be empty");
    }

    auto vertex_info = graph_info_->GetVertexInfo(vertex_type);
    if (!vertex_info) {
      return Status::Invalid("Vertex type '", vertex_type, "' not found");
    }
    for (const auto& prop : properties) {
      if (!vertex_info->HasProperty(prop)) {
        return Status::Invalid("Property '", prop,
                               "' not found in vertex type '", vertex_type,
                               "'");
      }
    }

    GAR_ASSIGN_OR_RAISE(
        auto chunk_count,
        util::GetVertexChunkNum(graph_info_->GetPrefix(), vertex_info));

    std::vector<FeatureGroupPlan> groups;
    std::unordered_map<PropertyGroup*, size_t> group_indices;
    for (const auto& prop : properties) {
      auto property_group = vertex_info->GetPropertyGroup(prop);
      if (!property_group) {
        return Status::Invalid("Property group not found for property '", prop,
                               "'");
      }
      auto [it, inserted] =
          group_indices.emplace(property_group.get(), groups.size());
      if (inserted) {
        groups.push_back(FeatureGroupPlan{property_group, {}});
      }
      auto& group_props = groups[it->second].properties;
      if (std::find(group_props.begin(), group_props.end(), prop) ==
          group_props.end()) {
        group_props.push_back(prop);
      }
    }

    const IdType chunk_size = vertex_info->GetChunkSize();
    std::map<IdType, std::vector<size_t>> chunk_to_indices;
    for (size_t i = 0; i < node_ids.size(); ++i) {
      if (node_ids[i] < 0) {
        return Status::IndexError("Negative node id ", node_ids[i]);
      }
      const IdType chunk_id = node_ids[i] / chunk_size;
      if (chunk_id >= chunk_count) {
        return Status::IndexError("Node id ", node_ids[i],
                                  " is outside vertex chunks");
      }
      chunk_to_indices[chunk_id].push_back(i);
    }

    request->request_id = next_request_id_.fetch_add(1);
    request->vertex_info = vertex_info;
    request->requested_properties = properties;
    request->row_count = node_ids.size();
    request->pending_chunks = groups.size() * chunk_to_indices.size();

    for (const auto& group : groups) {
      for (const auto& [chunk_id, indices] : chunk_to_indices) {
        FeatureChunkKey key;
        key.vertex_type = vertex_type;
        key.property_group = group.property_group;
        key.property_group_prefix = group.property_group->GetPrefix();
        key.file_type = group.property_group->GetFileType();
        key.chunk_id = chunk_id;

        FeatureChunkRows rows;
        rows.row_offsets.reserve(indices.size());
        rows.output_positions.reserve(indices.size());
        const IdType chunk_start = chunk_id * chunk_size;
        for (const size_t output_pos : indices) {
          rows.row_offsets.push_back(node_ids[output_pos] - chunk_start);
          rows.output_positions.push_back(static_cast<int64_t>(output_pos));
        }

        request->rows_by_key.emplace(key, std::move(rows));
        request->properties_by_key.emplace(key, group.properties);
      }
    }

    std::vector<std::pair<FeatureChunkKey, TablePtr>> trail_hits;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        return Status::Invalid("FeatureScanCoordinator is shut down");
      }
      GAR_RETURN_NOT_OK(EnsureStartedLocked(vertex_type, chunk_count));
      RegisterRequestLocked();
      for (const auto& [key, rows] : request->rows_by_key) {
        auto trail_table = LookupTrailLocked(key);
        if (trail_table != nullptr) {
          trail_hits_.fetch_add(1, std::memory_order_relaxed);
          trail_hits.emplace_back(key, trail_table);
        } else {
          trail_misses_.fetch_add(1, std::memory_order_relaxed);
          active_[key].push_back(request);
        }
      }
    }
    cv_.notify_all();

    for (const auto& [key, table] : trail_hits) {
      auto result = ServeRequestChunk(request, key, table);
      if (result.has_error()) {
        FailRequest(request, result.status());
        break;
      }
    }

    return handle;
  }

  FeatureCursorStats stats() const {
    FeatureCursorStats stats;
    stats.cursor_count =
        effective_cursor_count_.load(std::memory_order_relaxed);
    stats.trail_capacity_chunks = options_.trail_capacity_chunks;
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
    std::vector<RequestPtr> pending_requests;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        return;
      }
      shutdown_ = true;
      pending_requests = DrainActiveRequestsLocked();
    }
    cv_.notify_all();
    for (const auto& request : pending_requests) {
      FailRequest(
          request,
          Status::UnknownError(
              "FeatureScanCoordinator shutdown before request completion"));
    }
    for (auto& cursor : cursors_) {
      if (cursor.joinable()) {
        cursor.join();
      }
    }
  }

 private:
  void RegisterRequestLocked() {
    requests_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t active =
        active_requests_current_.fetch_add(1, std::memory_order_relaxed) + 1;
    AtomicMax(&active_requests_peak_, active);
  }

  std::vector<RequestPtr> DrainActiveRequestsLocked() {
    std::vector<RequestPtr> requests;
    std::unordered_set<FeatureRequestState*> seen;
    for (auto& [key, chunk_requests] : active_) {
      for (auto& request : chunk_requests) {
        if (request != nullptr && seen.insert(request.get()).second) {
          requests.push_back(request);
        }
      }
    }
    active_.clear();
    return requests;
  }

  Status EnsureStartedLocked(const std::string& vertex_type,
                             IdType chunk_count) {
    if (started_) {
      if (vertex_type != cursor_vertex_type_) {
        return Status::Invalid(
            "FeatureScanCoordinator only supports one vertex type per "
            "instance");
      }
      return Status::OK();
    }
    if (chunk_count <= 0) {
      return Status::Invalid("Vertex type '", vertex_type,
                             "' has no feature chunks");
    }

    cursor_vertex_type_ = vertex_type;
    cursor_chunk_count_ = chunk_count;
    const size_t cursor_count = std::min<size_t>(
        options_.cursor_count, static_cast<size_t>(chunk_count));
    effective_cursor_count_.store(cursor_count, std::memory_order_relaxed);
    started_ = true;

    cursors_.reserve(cursor_count);
    for (size_t i = 0; i < cursor_count; ++i) {
      const IdType begin = static_cast<IdType>(
          i * static_cast<size_t>(chunk_count) / cursor_count);
      const IdType end = static_cast<IdType>(
          (i + 1) * static_cast<size_t>(chunk_count) / cursor_count);
      cursors_.emplace_back([this, begin, end]() { CursorLoop(begin, end); });
    }
    return Status::OK();
  }

  bool HasActiveInRangeLocked(IdType begin, IdType end) const {
    for (const auto& [key, requests] : active_) {
      if (!requests.empty() && key.chunk_id >= begin && key.chunk_id < end) {
        return true;
      }
    }
    return false;
  }

  std::vector<FeatureChunkKey> ActiveKeysForChunk(IdType chunk_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<FeatureChunkKey> keys;
    for (const auto& [key, requests] : active_) {
      if (!requests.empty() && key.chunk_id == chunk_id) {
        keys.push_back(key);
      }
    }
    return keys;
  }

  void CursorLoop(IdType begin, IdType end) {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() {
          return shutdown_ || HasActiveInRangeLocked(begin, end);
        });
        if (shutdown_) {
          return;
        }
      }

      for (IdType chunk_id = begin; chunk_id < end; ++chunk_id) {
        if (IsShutdown()) {
          return;
        }
        auto keys = ActiveKeysForChunk(chunk_id);
        for (const auto& key : keys) {
          if (IsShutdown()) {
            return;
          }
          ServeCursorKey(key);
        }
      }
    }
  }

  bool IsShutdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_;
  }

  void ServeCursorKey(const FeatureChunkKey& key) {
    TableResult chunk_result;
    try {
      chunk_result = chunk_manager_->GetVertexPropertyChunk(
          graph_info_, key.vertex_type, key.property_group, key.chunk_id);
    } catch (const std::exception& e) {
      chunk_result =
          Status::UnknownError("Feature cursor chunk read threw: ", e.what());
    } catch (...) {
      chunk_result =
          Status::UnknownError("Feature cursor chunk read threw unknown error");
    }
    chunks_read_.fetch_add(1, std::memory_order_relaxed);

    if (chunk_result.has_error()) {
      auto requests = TakeActiveRequests(key);
      for (const auto& request : requests) {
        FailRequest(request, chunk_result.status());
      }
      return;
    }

    auto table = chunk_result.value();
    auto requests = TakeActiveRequestsAndInsertTrail(key, table);
    if (requests.empty()) {
      return;
    }

    chunks_served_.fetch_add(1, std::memory_order_relaxed);
    const auto service_start = Clock::now();
    for (const auto& request : requests) {
      auto result = ServeRequestChunk(request, key, table);
      if (result.has_error()) {
        FailRequest(request, result.status());
      }
    }
    const uint64_t service_ms = MillisecondsSince(service_start);
    service_ms_sum_.fetch_add(service_ms, std::memory_order_relaxed);
    AtomicMax(&service_ms_max_, service_ms);
  }

  std::vector<RequestPtr> TakeActiveRequests(const FeatureChunkKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_.find(key);
    if (it == active_.end()) {
      return {};
    }
    auto requests = std::move(it->second);
    active_.erase(it);
    return requests;
  }

  std::vector<RequestPtr> TakeActiveRequestsAndInsertTrail(
      const FeatureChunkKey& key, const TablePtr& table) {
    std::lock_guard<std::mutex> lock(mutex_);
    InsertTrailLocked(key, table);
    auto it = active_.find(key);
    if (it == active_.end()) {
      return {};
    }
    auto requests = std::move(it->second);
    active_.erase(it);
    return requests;
  }

  TableResult ServeRequestChunk(const RequestPtr& request,
                                const FeatureChunkKey& key,
                                const TablePtr& table) {
    FeatureChunkRows rows;
    std::vector<std::string> properties;
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      if (request->completed) {
        return table;
      }
      auto rows_it = request->rows_by_key.find(key);
      auto props_it = request->properties_by_key.find(key);
      if (rows_it == request->rows_by_key.end() ||
          props_it == request->properties_by_key.end()) {
        return table;
      }
      rows = rows_it->second;
      properties = props_it->second;
    }

    std::vector<int64_t> row_offsets;
    row_offsets.reserve(rows.row_offsets.size());
    for (const auto row_offset : rows.row_offsets) {
      row_offsets.push_back(row_offset);
    }
    auto take_indices = MakeInt64Array(row_offsets);
    if (take_indices == nullptr) {
      return Status::ArrowError("Failed to build feature take indices");
    }

    std::unordered_map<std::string, std::shared_ptr<arrow::Array>>
        taken_by_property;
    for (const auto& prop : properties) {
      auto column = table->GetColumnByName(prop);
      if (!column) {
        return Status::Invalid("Column '", prop, "' not found in chunk");
      }
      auto combined_result = arrow::Table::Make(
          arrow::schema({arrow::field(prop, column->type())}), {column})
                                 ->CombineChunks();
      if (!combined_result.ok()) {
        return Status::ArrowError(combined_result.status().ToString());
      }
      auto combined_column = combined_result.ValueOrDie()->column(0);
      if (combined_column->num_chunks() == 0) {
        return Status::Invalid("Column '", prop, "' has no chunks");
      }
      auto take_result =
          arrow::compute::Take(combined_column->chunk(0), take_indices);
      if (!take_result.ok()) {
        return Status::ArrowError(take_result.status().ToString());
      }
      taken_by_property[prop] = take_result.ValueOrDie().make_array();
    }

    TableResult completion;
    bool should_complete = false;
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      if (request->completed) {
        return table;
      }
      for (const auto& prop : properties) {
        request->arrays_by_property[prop].push_back(taken_by_property[prop]);
        auto& positions = request->positions_by_property[prop];
        positions.insert(positions.end(), rows.output_positions.begin(),
                         rows.output_positions.end());
      }
      rows_served_.fetch_add(rows.output_positions.size(),
                             std::memory_order_relaxed);
      batches_served_.fetch_add(1, std::memory_order_relaxed);

      if (request->pending_chunks == 0) {
        return Status::Invalid("Feature request pending chunk underflow");
      }
      --request->pending_chunks;
      if (request->pending_chunks == 0) {
        completion = BuildFinalTableLocked(*request);
        request->completed = true;
        should_complete = true;
      }
    }

    if (should_complete) {
      CompleteRequest(request, std::move(completion));
    }
    return table;
  }

  void FailRequest(const RequestPtr& request, const Status& status) {
    bool should_complete = false;
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      if (!request->completed) {
        request->completed = true;
        should_complete = true;
      }
    }
    if (should_complete) {
      TableResult result = status;
      CompleteRequest(request, std::move(result));
    }
  }

  void CompleteRequest(const RequestPtr& request, TableResult result) {
    const uint64_t wait_ms = MillisecondsSince(request->submit_time);
    wait_ms_sum_.fetch_add(wait_ms, std::memory_order_relaxed);
    AtomicMax(&wait_ms_max_, wait_ms);
    active_requests_current_.fetch_sub(1, std::memory_order_relaxed);
    if (result.has_error()) {
      requests_failed_.fetch_add(1, std::memory_order_relaxed);
    } else {
      requests_completed_.fetch_add(1, std::memory_order_relaxed);
    }
    request->promise.set_value(std::move(result));
  }

  TablePtr LookupTrailLocked(const FeatureChunkKey& key) const {
    auto it = trail_index_.find(key);
    if (it == trail_index_.end()) {
      return nullptr;
    }
    return it->second->table;
  }

  void InsertTrailLocked(const FeatureChunkKey& key, const TablePtr& table) {
    if (options_.trail_capacity_chunks == 0 || table == nullptr) {
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
    while (trail_.size() > options_.trail_capacity_chunks) {
      trail_index_.erase(trail_.front().key);
      trail_.pop_front();
      trail_evictions_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  struct TrailEntry {
    FeatureChunkKey key;
    TablePtr table;
  };

  std::shared_ptr<GraphInfo> graph_info_;
  std::shared_ptr<ChunkReadManager> chunk_manager_;
  FeatureCursorOptions options_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool shutdown_ = false;
  bool started_ = false;
  std::string cursor_vertex_type_;
  IdType cursor_chunk_count_ = 0;
  std::vector<std::thread> cursors_;
  std::unordered_map<FeatureChunkKey, std::vector<RequestPtr>,
                     FeatureChunkKeyHash>
      active_;
  std::list<TrailEntry> trail_;
  std::unordered_map<FeatureChunkKey, std::list<TrailEntry>::iterator,
                     FeatureChunkKeyHash>
      trail_index_;

  std::atomic<uint64_t> next_request_id_{1};
  std::atomic<uint64_t> effective_cursor_count_{0};
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

FeatureScanCoordinator::FeatureScanCoordinator(
    std::shared_ptr<GraphInfo> graph_info,
    std::shared_ptr<ChunkReadManager> chunk_manager,
    FeatureCursorOptions options)
    : impl_(std::make_unique<Impl>(std::move(graph_info),
                                   std::move(chunk_manager), options)) {}

FeatureScanCoordinator::~FeatureScanCoordinator() = default;

FeatureScanCoordinator::HandleResult FeatureScanCoordinator::Submit(
    const std::string& vertex_type, const std::vector<IdType>& node_ids,
    const std::vector<std::string>& properties) {
  return impl_->Submit(vertex_type, node_ids, properties);
}

FeatureCursorStats FeatureScanCoordinator::stats() const {
  return impl_->stats();
}

void FeatureScanCoordinator::Shutdown() { impl_->Shutdown(); }

}  // namespace graphar::ml
