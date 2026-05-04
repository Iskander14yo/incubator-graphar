#include "graphar/ml/feature_pipeline.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/compute/api.h"
#include "graphar/graph_info.h"
#include "graphar/ml/chunk_read_manager.h"
#include "graphar/status.h"
#include "graphar/types.h"

namespace graphar::ml {
namespace {

using Clock = std::chrono::steady_clock;

template <typename T>
void AtomicMax(std::atomic<T>* target, T value) {
  T current = target->load(std::memory_order_relaxed);
  while (current < value &&
         !target->compare_exchange_weak(current, value,
                                        std::memory_order_relaxed)) {
  }
}

uint64_t MillisecondsBetween(Clock::time_point start, Clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count());
}

FeaturePipelineOptions NormalizeOptions(FeaturePipelineOptions options) {
  options.num_readers = std::max<size_t>(1, options.num_readers);
  options.num_stitchers = std::max<size_t>(1, options.num_stitchers);
  options.max_active_batches = std::max<size_t>(1, options.max_active_batches);
  options.max_queued_stitch_tasks =
      std::max(options.max_queued_stitch_tasks, options.num_stitchers);
  return options;
}

ChunkReadKey MakeFeatureChunkKey(
    const std::shared_ptr<GraphInfo>& graph_info, const std::string& vertex_type,
    const std::shared_ptr<PropertyGroup>& property_group, IdType chunk_id) {
  ChunkReadKey key;
  key.kind = ChunkReadKind::kVertexProperty;
  key.graph_prefix = graph_info->GetPrefix();
  key.vertex_type = vertex_type;
  key.property_group_prefix = property_group->GetPrefix();
  key.file_type = property_group->GetFileType();
  key.chunk_id = chunk_id;
  return key;
}

std::shared_ptr<arrow::Table> MakeEmptyTable() {
  return arrow::Table::Make(arrow::schema({}),
                            std::vector<std::shared_ptr<arrow::Array>>{}, 0);
}

}  // namespace

struct FeatureBatchState {
  using TablePtr = std::shared_ptr<arrow::Table>;
  using TableResult = Result<TablePtr>;

  uint64_t id = 0;
  std::shared_ptr<GraphInfo> graph_info;
  std::string vertex_type;
  std::shared_ptr<VertexInfo> vertex_info;
  std::vector<std::string> requested_properties;
  std::unordered_map<std::string, size_t> property_index;
  std::vector<std::vector<std::shared_ptr<arrow::Array>>> property_arrays;
  std::vector<std::vector<int64_t>> property_positions;
  size_t row_count = 0;
  size_t pending_chunks = 0;
  Clock::time_point submitted_at = Clock::now();
  Clock::time_point ready_at = Clock::now();
  bool feature_request_registered = false;
  std::promise<TableResult> promise;
  std::shared_future<TableResult> future = promise.get_future().share();

  mutable std::mutex mutex_;
  bool promise_set = false;
};

struct FeaturePipelineCoordinator::Impl {
  using TablePtr = std::shared_ptr<arrow::Table>;
  using TableResult = Result<TablePtr>;

  struct Subscription {
    std::shared_ptr<FeatureBatchState> batch;
    std::vector<std::string> properties;
    std::vector<int64_t> positions;
    std::vector<int64_t> row_ids;
  };

  struct SubscriptionRequest {
    std::shared_ptr<PropertyGroup> property_group;
    IdType chunk_id = 0;
    std::vector<std::string> properties;
    std::vector<int64_t> positions;
    std::vector<int64_t> row_ids;
  };

  struct ActiveChunk {
    ChunkReadKey key;
    std::shared_ptr<GraphInfo> graph_info;
    std::string vertex_type;
    std::shared_ptr<PropertyGroup> property_group;
    IdType chunk_id = 0;
    TablePtr table;
    Status error = Status::UnknownError("chunk not loaded");
    bool ready = false;
    bool failed = false;
    size_t in_flight_stitches = 0;
    std::vector<Subscription> waiting_subscribers;
  };

  struct StitchTask {
    std::shared_ptr<ActiveChunk> active_chunk;
    Subscription subscription;
    TablePtr table;
    Clock::time_point enqueued_at = Clock::now();
  };

  Impl(std::shared_ptr<ChunkReadManager> chunk_manager,
       FeaturePipelineOptions options)
      : chunk_manager_(std::move(chunk_manager)),
        options_(NormalizeOptions(options)) {
    reader_threads_.reserve(options_.num_readers);
    for (size_t i = 0; i < options_.num_readers; ++i) {
      reader_threads_.emplace_back([this]() { ReaderLoop(); });
    }
    stitcher_threads_.reserve(options_.num_stitchers);
    for (size_t i = 0; i < options_.num_stitchers; ++i) {
      stitcher_threads_.emplace_back([this]() { StitcherLoop(); });
    }
  }

  ~Impl() { Shutdown(); }

  Result<std::shared_ptr<FeatureBatchHandle>> SubmitSampledBatch(
      const std::shared_ptr<GraphInfo>& graph_info,
      const std::string& vertex_type, const std::vector<IdType>& node_ids,
      const std::vector<std::string>& properties) {
    if (chunk_manager_ == nullptr) {
      return Status::Invalid("Feature pipeline requires a chunk manager");
    }
    if (graph_info == nullptr) {
      return Status::Invalid("GraphInfo cannot be null");
    }
    if (properties.empty()) {
      return Status::Invalid("Properties list cannot be empty");
    }

    auto vertex_info = graph_info->GetVertexInfo(vertex_type);
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

    auto batch = std::make_shared<FeatureBatchState>();
    batch->id = next_batch_id_.fetch_add(1, std::memory_order_relaxed);
    batch->graph_info = graph_info;
    batch->vertex_type = vertex_type;
    batch->vertex_info = vertex_info;
    batch->requested_properties = properties;
    batch->row_count = node_ids.size();
    batch->submitted_at = Clock::now();
    chunk_manager_->RegisterFeatureRequest();
    batch->feature_request_registered = true;

    for (const auto& prop : properties) {
      if (batch->property_index.find(prop) != batch->property_index.end()) {
        continue;
      }
      const size_t next_index = batch->property_arrays.size();
      batch->property_index.emplace(prop, next_index);
      batch->property_arrays.emplace_back();
      batch->property_positions.emplace_back();
    }

    submitted_batches_.fetch_add(1, std::memory_order_relaxed);

    {
      std::unique_lock<std::mutex> lock(mutex_);
      active_batches_cv_.wait(lock, [&]() {
        return shutdown_ ||
               active_batches_.size() < options_.max_active_batches;
      });
      if (shutdown_) {
        lock.unlock();
        chunk_manager_->CompleteFeatureRequest(batch->submitted_at, false);
        return Status::Invalid("Feature pipeline is shut down");
      }
      active_batches_.emplace(batch->id, batch);
      AtomicMax(&pending_batches_peak_,
                static_cast<uint64_t>(active_batches_.size()));
    }

    if (node_ids.empty()) {
      FinishBatch(batch, MakeEmptyTable());
      return std::make_shared<FeatureBatchHandle>(batch);
    }

    std::unordered_map<std::shared_ptr<PropertyGroup>, std::vector<std::string>>
        pg_to_props;
    for (const auto& prop : properties) {
      auto property_group = vertex_info->GetPropertyGroup(prop);
      if (!property_group) {
        FinishBatch(
            batch,
            Status::Invalid("Property group not found for property '", prop,
                            "'"));
        return std::make_shared<FeatureBatchHandle>(batch);
      }
      pg_to_props[property_group].push_back(prop);
    }

    std::vector<SubscriptionRequest> requests;
    for (const auto& [property_group, group_properties] : pg_to_props) {
      const IdType chunk_size = vertex_info->GetChunkSize();
      std::map<IdType, std::vector<size_t>> chunk_to_indices;
      for (size_t i = 0; i < node_ids.size(); ++i) {
        chunk_to_indices[node_ids[i] / chunk_size].push_back(i);
      }

      for (const auto& [chunk_id, indices] : chunk_to_indices) {
        SubscriptionRequest request;
        request.property_group = property_group;
        request.chunk_id = chunk_id;
        request.properties = group_properties;
        request.positions.reserve(indices.size());
        request.row_ids.reserve(indices.size());

        const IdType chunk_start = chunk_id * chunk_size;
        for (size_t index : indices) {
          request.positions.push_back(static_cast<int64_t>(index));
          request.row_ids.push_back(
              static_cast<int64_t>(node_ids[index] - chunk_start));
        }
        requests.push_back(std::move(request));
      }
    }

    {
      std::lock_guard<std::mutex> lock(batch->mutex_);
      batch->pending_chunks = requests.size();
    }

    if (requests.empty()) {
      FinishBatch(batch, MakeEmptyTable());
      return std::make_shared<FeatureBatchHandle>(batch);
    }

    for (const auto& request : requests) {
      auto status = AddSubscription(batch, request);
      if (!status.ok()) {
        FinishBatch(batch, status);
        break;
      }
    }

    return std::make_shared<FeatureBatchHandle>(batch);
  }

  FeaturePipelineStats Stats() const {
    FeaturePipelineStats stats;
    stats.submitted_batches =
        submitted_batches_.load(std::memory_order_relaxed);
    stats.completed_batches =
        completed_batches_.load(std::memory_order_relaxed);
    stats.pending_batches_peak =
        pending_batches_peak_.load(std::memory_order_relaxed);
    stats.active_chunk_keys_peak =
        active_chunk_keys_peak_.load(std::memory_order_relaxed);
    stats.chunk_subscriptions =
        chunk_subscriptions_.load(std::memory_order_relaxed);
    stats.chunk_reads = chunk_reads_.load(std::memory_order_relaxed);
    stats.chunk_reuses = chunk_reuses_.load(std::memory_order_relaxed);
    stats.stitch_tasks = stitch_tasks_.load(std::memory_order_relaxed);
    stats.stitch_wait_ms_sum =
        stitch_wait_ms_sum_.load(std::memory_order_relaxed);
    stats.stitch_service_ms_sum =
        stitch_service_ms_sum_.load(std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stats.active_batches_current = active_batches_.size();
      stats.active_chunk_keys_current = active_chunks_.size();
      stats.read_queue_current = read_queue_.size();
      stats.stitch_queue_current = stitch_queue_.size();
    }
    return stats;
  }

  void Shutdown() {
    std::vector<std::shared_ptr<FeatureBatchState>> batches_to_fail;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        return;
      }
      shutdown_ = true;
      for (const auto& [_, batch] : active_batches_) {
        batches_to_fail.push_back(batch);
      }
      read_queue_.clear();
      stitch_queue_.clear();
      active_chunks_.clear();
    }

    active_batches_cv_.notify_all();
    read_cv_.notify_all();
    stitch_cv_.notify_all();
    stitch_space_cv_.notify_all();

    for (const auto& batch : batches_to_fail) {
      FinishBatch(batch, Status::Invalid("Feature pipeline is shut down"));
    }

    for (auto& thread : reader_threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    reader_threads_.clear();

    for (auto& thread : stitcher_threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    stitcher_threads_.clear();
  }

 private:
  Status AddSubscription(const std::shared_ptr<FeatureBatchState>& batch,
                         const SubscriptionRequest& request) {
    Subscription subscription;
    subscription.batch = batch;
    subscription.properties = request.properties;
    subscription.positions = request.positions;
    subscription.row_ids = request.row_ids;

    ChunkReadKey key =
        MakeFeatureChunkKey(batch->graph_info, batch->vertex_type,
                            request.property_group, request.chunk_id);

    std::shared_ptr<ActiveChunk> active_chunk;
    TablePtr ready_table;
    std::unique_ptr<Status> chunk_error;
    bool enqueue_read = false;
    bool enqueue_stitch = false;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (shutdown_) {
        return Status::Invalid("Feature pipeline is shut down");
      }

      chunk_subscriptions_.fetch_add(1, std::memory_order_relaxed);

      auto it = active_chunks_.find(key);
      if (it == active_chunks_.end()) {
        active_chunk = std::make_shared<ActiveChunk>();
        active_chunk->key = key;
        active_chunk->graph_info = batch->graph_info;
        active_chunk->vertex_type = batch->vertex_type;
        active_chunk->property_group = request.property_group;
        active_chunk->chunk_id = request.chunk_id;
        active_chunk->waiting_subscribers.push_back(std::move(subscription));
        active_chunks_.emplace(key, active_chunk);
        AtomicMax(&active_chunk_keys_peak_,
                  static_cast<uint64_t>(active_chunks_.size()));
        read_queue_.push_back(active_chunk);
        enqueue_read = true;
      } else {
        active_chunk = it->second;
        chunk_reuses_.fetch_add(1, std::memory_order_relaxed);
        if (active_chunk->failed) {
          chunk_error = std::make_unique<Status>(active_chunk->error);
        } else if (active_chunk->ready) {
          active_chunk->in_flight_stitches += 1;
          ready_table = active_chunk->table;
          enqueue_stitch = true;
        } else {
          active_chunk->waiting_subscribers.push_back(std::move(subscription));
        }
      }
    }

    if (enqueue_read) {
      read_cv_.notify_one();
      return Status::OK();
    }
    if (chunk_error != nullptr) {
      return std::move(*chunk_error);
    }
    if (!enqueue_stitch) {
      return Status::OK();
    }
    return EnqueueStitchTask(active_chunk, std::move(subscription), ready_table);
  }

  Status EnqueueStitchTask(const std::shared_ptr<ActiveChunk>& active_chunk,
                           Subscription subscription, const TablePtr& table) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      stitch_space_cv_.wait(lock, [&]() {
        return shutdown_ ||
               stitch_queue_.size() < options_.max_queued_stitch_tasks;
      });
      if (shutdown_) {
        if (active_chunk != nullptr && active_chunk->in_flight_stitches > 0) {
          active_chunk->in_flight_stitches -= 1;
        }
        return Status::Invalid("Feature pipeline is shut down");
      }
      stitch_queue_.push_back(StitchTask{
          active_chunk, std::move(subscription), table, Clock::now()});
      stitch_tasks_.fetch_add(1, std::memory_order_relaxed);
    }
    stitch_cv_.notify_one();
    return Status::OK();
  }

  void ReaderLoop() {
    while (true) {
      std::shared_ptr<ActiveChunk> active_chunk;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        read_cv_.wait(lock,
                      [&]() { return shutdown_ || !read_queue_.empty(); });
        if (shutdown_ && read_queue_.empty()) {
          return;
        }
        active_chunk = read_queue_.front();
        read_queue_.pop_front();
      }

      if (active_chunk == nullptr) {
        continue;
      }

      auto result = chunk_manager_->GetVertexPropertyChunk(
          active_chunk->graph_info, active_chunk->vertex_type,
          active_chunk->property_group, active_chunk->chunk_id);
      chunk_reads_.fetch_add(1, std::memory_order_relaxed);

      std::vector<Subscription> subscriptions;
      std::shared_ptr<ActiveChunk> current_chunk;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = active_chunks_.find(active_chunk->key);
        if (it == active_chunks_.end()) {
          continue;
        }
        current_chunk = it->second;
        subscriptions = std::move(current_chunk->waiting_subscribers);
        if (shutdown_) {
          active_chunks_.erase(it);
        } else if (result.has_error()) {
          current_chunk->failed = true;
          current_chunk->error = result.status();
          active_chunks_.erase(it);
        } else {
          current_chunk->ready = true;
          current_chunk->table = result.value();
          current_chunk->in_flight_stitches += subscriptions.size();
          if (subscriptions.empty() &&
              current_chunk->in_flight_stitches == 0) {
            active_chunks_.erase(it);
          }
        }
      }

      if (shutdown_) {
        FailSubscriptions(subscriptions,
                          Status::Invalid("Feature pipeline is shut down"));
        continue;
      }

      if (result.has_error()) {
        FailSubscriptions(subscriptions, result.status());
        continue;
      }

      for (auto& subscription : subscriptions) {
        auto batch = subscription.batch;
        auto status = EnqueueStitchTask(current_chunk, std::move(subscription),
                                        result.value());
        if (!status.ok()) {
          FailBatch(batch, status);
        }
      }
    }
  }

  Result<std::vector<std::shared_ptr<arrow::Array>>> StitchRows(
      const TablePtr& table, const std::vector<std::string>& properties,
      const std::vector<int64_t>& row_ids) const {
    arrow::Int64Builder idx_builder;
    auto status = idx_builder.Reserve(row_ids.size());
    if (!status.ok()) {
      return Status::ArrowError(status.ToString());
    }
    for (int64_t row_id : row_ids) {
      idx_builder.UnsafeAppend(row_id);
    }

    std::shared_ptr<arrow::Array> take_indices;
    status = idx_builder.Finish(&take_indices);
    if (!status.ok()) {
      return Status::ArrowError(status.ToString());
    }

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(properties.size());
    for (const auto& prop : properties) {
      auto column = table->GetColumnByName(prop);
      if (!column) {
        return Status::Invalid("Column '", prop, "' not found in chunk");
      }
      auto combined_result = arrow::Table::Make(
                                 arrow::schema(
                                     {arrow::field(prop, column->type())}),
                                 {column})
                                 ->CombineChunks();
      if (!combined_result.ok()) {
        return Status::ArrowError(combined_result.status().ToString());
      }
      auto combined_column = combined_result.ValueOrDie()->column(0);
      auto take_result =
          arrow::compute::Take(combined_column->chunk(0), take_indices);
      if (!take_result.ok()) {
        return Status::ArrowError(take_result.status().ToString());
      }
      arrays.push_back(take_result.ValueOrDie().make_array());
    }
    return arrays;
  }

  TableResult BuildBatchTable(
      const std::shared_ptr<FeatureBatchState>& batch) const {
    std::vector<std::vector<std::shared_ptr<arrow::Array>>> property_arrays;
    std::vector<std::vector<int64_t>> property_positions;
    std::vector<std::string> requested_properties;
    std::unordered_map<std::string, size_t> property_index;
    std::shared_ptr<VertexInfo> vertex_info;
    size_t row_count = 0;

    {
      std::lock_guard<std::mutex> lock(batch->mutex_);
      property_arrays = batch->property_arrays;
      property_positions = batch->property_positions;
      requested_properties = batch->requested_properties;
      property_index = batch->property_index;
      vertex_info = batch->vertex_info;
      row_count = batch->row_count;
    }

    std::unordered_map<std::string, std::shared_ptr<arrow::Array>>
        result_arrays_by_property;
    result_arrays_by_property.reserve(property_index.size());

    for (const auto& [prop, index] : property_index) {
      if (property_arrays[index].empty()) {
        return Status::Invalid("No feature data collected for property '", prop,
                               "'");
      }
      if (property_positions[index].size() != row_count) {
        return Status::Invalid(
            "Incomplete feature data collected for property '", prop, "'");
      }

      auto concat_result = arrow::Concatenate(property_arrays[index]);
      if (!concat_result.ok()) {
        return Status::ArrowError(concat_result.status().ToString());
      }

      std::vector<int64_t> inv_perm(row_count);
      for (size_t i = 0; i < property_positions[index].size(); ++i) {
        inv_perm[property_positions[index][i]] = static_cast<int64_t>(i);
      }

      arrow::Int64Builder perm_builder;
      auto status = perm_builder.AppendValues(inv_perm);
      if (!status.ok()) {
        return Status::ArrowError(status.ToString());
      }

      std::shared_ptr<arrow::Array> perm_array;
      status = perm_builder.Finish(&perm_array);
      if (!status.ok()) {
        return Status::ArrowError(status.ToString());
      }

      auto reorder_result =
          arrow::compute::Take(concat_result.ValueOrDie(), perm_array);
      if (!reorder_result.ok()) {
        return Status::ArrowError(reorder_result.status().ToString());
      }
      result_arrays_by_property[prop] = reorder_result.ValueOrDie().make_array();
    }

    std::vector<std::shared_ptr<arrow::Field>> schema_fields;
    std::vector<std::shared_ptr<arrow::Array>> result_arrays;
    schema_fields.reserve(requested_properties.size());
    result_arrays.reserve(requested_properties.size());

    for (const auto& prop : requested_properties) {
      auto prop_type_result = vertex_info->GetPropertyType(prop);
      GAR_RETURN_NOT_OK(prop_type_result.status());
      auto arrow_type =
          DataType::DataTypeToArrowDataType(prop_type_result.value());
      schema_fields.push_back(arrow::field(prop, arrow_type));
      result_arrays.push_back(result_arrays_by_property.at(prop));
    }

    return arrow::Table::Make(arrow::schema(schema_fields), result_arrays,
                              row_count);
  }

  void StitcherLoop() {
    while (true) {
      StitchTask task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        stitch_cv_.wait(lock,
                        [&]() { return shutdown_ || !stitch_queue_.empty(); });
        if (shutdown_ && stitch_queue_.empty()) {
          return;
        }
        task = std::move(stitch_queue_.front());
        stitch_queue_.pop_front();
      }
      stitch_space_cv_.notify_all();

      const auto service_start = Clock::now();
      const auto wait_ms = MillisecondsBetween(task.enqueued_at, service_start);
      stitch_wait_ms_sum_.fetch_add(wait_ms, std::memory_order_relaxed);

      auto arrays_result =
          StitchRows(task.table, task.subscription.properties,
                     task.subscription.row_ids);
      if (arrays_result.has_error()) {
        FailBatch(task.subscription.batch, arrays_result.status());
        FinishChunkTask(task.active_chunk);
        continue;
      }

      bool finalize_batch = false;
      {
        std::lock_guard<std::mutex> lock(task.subscription.batch->mutex_);
        if (!task.subscription.batch->promise_set) {
          chunk_manager_->RecordFeatureBatchServed(task.subscription.positions.size());
          for (size_t i = 0; i < task.subscription.properties.size(); ++i) {
            const auto index = task.subscription.batch->property_index.at(
                task.subscription.properties[i]);
            task.subscription.batch->property_arrays[index].push_back(
                arrays_result.value()[i]);
            auto& positions =
                task.subscription.batch->property_positions[index];
            positions.insert(positions.end(),
                             task.subscription.positions.begin(),
                             task.subscription.positions.end());
          }
          if (task.subscription.batch->pending_chunks > 0) {
            task.subscription.batch->pending_chunks -= 1;
          }
          finalize_batch = task.subscription.batch->pending_chunks == 0;
        }
      }

      if (finalize_batch) {
        FinishBatch(task.subscription.batch,
                    BuildBatchTable(task.subscription.batch));
      }

      const auto service_ms =
          MillisecondsBetween(service_start, Clock::now());
      stitch_service_ms_sum_.fetch_add(service_ms,
                                       std::memory_order_relaxed);
      FinishChunkTask(task.active_chunk);
    }
  }

  void FinishChunkTask(const std::shared_ptr<ActiveChunk>& active_chunk) {
    if (active_chunk == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_chunks_.find(active_chunk->key);
    if (it == active_chunks_.end()) {
      return;
    }
    if (it->second->in_flight_stitches > 0) {
      it->second->in_flight_stitches -= 1;
    }
    if (it->second->ready && it->second->waiting_subscribers.empty() &&
        it->second->in_flight_stitches == 0) {
      active_chunks_.erase(it);
    }
  }

  void FailSubscriptions(const std::vector<Subscription>& subscriptions,
                         const Status& status) {
    for (const auto& subscription : subscriptions) {
      FailBatch(subscription.batch, status);
    }
  }

  void FailBatch(const std::shared_ptr<FeatureBatchState>& batch,
                 const Status& status) {
    FinishBatch(batch, status);
  }

  void FinishBatch(const std::shared_ptr<FeatureBatchState>& batch,
                   const TableResult& result) {
    if (batch == nullptr) {
      return;
    }

    bool should_set = false;
    {
      std::lock_guard<std::mutex> lock(batch->mutex_);
      if (!batch->promise_set) {
        batch->promise_set = true;
        batch->ready_at = Clock::now();
        should_set = true;
      }
    }

    if (!should_set) {
      return;
    }

    batch->promise.set_value(result);
    if (batch->feature_request_registered) {
      chunk_manager_->CompleteFeatureRequest(batch->submitted_at,
                                            !result.has_error());
    }
    completed_batches_.fetch_add(1, std::memory_order_relaxed);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      active_batches_.erase(batch->id);
    }
    active_batches_cv_.notify_all();
  }

  std::shared_ptr<ChunkReadManager> chunk_manager_;
  FeaturePipelineOptions options_;

  mutable std::mutex mutex_;
  std::condition_variable active_batches_cv_;
  std::condition_variable read_cv_;
  std::condition_variable stitch_cv_;
  std::condition_variable stitch_space_cv_;
  bool shutdown_ = false;

  std::unordered_map<uint64_t, std::shared_ptr<FeatureBatchState>>
      active_batches_;
  std::unordered_map<ChunkReadKey, std::shared_ptr<ActiveChunk>,
                     ChunkReadKeyHash>
      active_chunks_;
  std::deque<std::shared_ptr<ActiveChunk>> read_queue_;
  std::deque<StitchTask> stitch_queue_;
  std::vector<std::thread> reader_threads_;
  std::vector<std::thread> stitcher_threads_;

  std::atomic<uint64_t> next_batch_id_{0};
  std::atomic<uint64_t> submitted_batches_{0};
  std::atomic<uint64_t> completed_batches_{0};
  std::atomic<uint64_t> pending_batches_peak_{0};
  std::atomic<uint64_t> active_chunk_keys_peak_{0};
  std::atomic<uint64_t> chunk_subscriptions_{0};
  std::atomic<uint64_t> chunk_reads_{0};
  std::atomic<uint64_t> chunk_reuses_{0};
  std::atomic<uint64_t> stitch_tasks_{0};
  std::atomic<uint64_t> stitch_wait_ms_sum_{0};
  std::atomic<uint64_t> stitch_service_ms_sum_{0};
};

FeatureBatchHandle::FeatureBatchHandle() = default;

FeatureBatchHandle::FeatureBatchHandle(std::shared_ptr<FeatureBatchState> state)
    : state_(std::move(state)) {}

FeatureBatchHandle::~FeatureBatchHandle() = default;

FeatureBatchHandle::FeatureBatchHandle(const FeatureBatchHandle&) = default;

FeatureBatchHandle& FeatureBatchHandle::operator=(const FeatureBatchHandle&) =
    default;

FeatureBatchHandle::FeatureBatchHandle(FeatureBatchHandle&&) noexcept = default;

FeatureBatchHandle& FeatureBatchHandle::operator=(
    FeatureBatchHandle&&) noexcept = default;

FeatureBatchHandle::TableResult FeatureBatchHandle::Wait() const {
  if (state_ == nullptr) {
    return Status::Invalid("Feature batch handle is empty");
  }
  return state_->future.get();
}

uint64_t FeatureBatchHandle::feature_fetch_ms() const {
  if (state_ == nullptr) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(state_->mutex_);
  if (!state_->promise_set) {
    return 0;
  }
  return MillisecondsBetween(state_->submitted_at, state_->ready_at);
}

bool FeatureBatchHandle::valid() const { return state_ != nullptr; }

FeaturePipelineCoordinator::FeaturePipelineCoordinator(
    std::shared_ptr<ChunkReadManager> chunk_manager,
    FeaturePipelineOptions options)
    : impl_(
          std::make_shared<Impl>(std::move(chunk_manager), std::move(options))) {
}

FeaturePipelineCoordinator::~FeaturePipelineCoordinator() = default;

FeaturePipelineCoordinator::FeaturePipelineCoordinator(
    const FeaturePipelineCoordinator&) = default;

FeaturePipelineCoordinator& FeaturePipelineCoordinator::operator=(
    const FeaturePipelineCoordinator&) = default;

FeaturePipelineCoordinator::FeaturePipelineCoordinator(
    FeaturePipelineCoordinator&&) noexcept = default;

FeaturePipelineCoordinator& FeaturePipelineCoordinator::operator=(
    FeaturePipelineCoordinator&&) noexcept = default;

Result<std::shared_ptr<FeatureBatchHandle>>
FeaturePipelineCoordinator::SubmitSampledBatch(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::string& vertex_type, const std::vector<IdType>& node_ids,
    const std::vector<std::string>& properties) {
  return impl_->SubmitSampledBatch(graph_info, vertex_type, node_ids,
                                   properties);
}

FeaturePipelineStats FeaturePipelineCoordinator::Stats() const {
  return impl_->Stats();
}

void FeaturePipelineCoordinator::Shutdown() { impl_->Shutdown(); }

}  // namespace graphar::ml
