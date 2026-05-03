#include "graphar/ml/edge_pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "graphar/graph_info.h"
#include "graphar/ml/chunk_read_manager.h"
#include "graphar/ml/neighbor_sampling.h"
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

EdgeSamplingPipelineOptions NormalizeOptions(EdgeSamplingPipelineOptions options) {
  options.num_readers = std::max<size_t>(1, options.num_readers);
  options.num_processors = std::max<size_t>(1, options.num_processors);
  options.max_active_batches = std::max<size_t>(1, options.max_active_batches);
  options.max_queued_processor_tasks =
      std::max(options.max_queued_processor_tasks, options.num_processors);
  return options;
}

void SampleUniformLocalIndices(IdType n, IdType k, std::mt19937* gen,
                               std::vector<IdType>* sorted_out) {
  sorted_out->clear();
  if (k == 0 || n == 0) {
    return;
  }
  std::unordered_set<IdType> chosen;
  chosen.reserve(static_cast<size_t>(k) * 2);
  for (IdType i = n - k; i < n; ++i) {
    std::uniform_int_distribution<IdType> dist(0, i);
    const IdType j = dist(*gen);
    if (chosen.find(j) != chosen.end()) {
      chosen.insert(i);
    } else {
      chosen.insert(j);
    }
  }
  sorted_out->assign(chosen.begin(), chosen.end());
  std::sort(sorted_out->begin(), sorted_out->end());
}

class Int64ChunkedArrayCursor {
 public:
  explicit Int64ChunkedArrayCursor(
      std::shared_ptr<arrow::ChunkedArray> chunked_array)
      : chunked_array_(std::move(chunked_array)) {
    Reset();
  }

  Status AdvanceTo(IdType row) {
    if (chunked_array_ == nullptr || chunked_array_->num_chunks() == 0) {
      return Status::Invalid("Chunked int64 column has no rows");
    }

    if (row < chunk_begin_) {
      Reset();
    }

    while (row >= chunk_begin_ + chunk_length_) {
      chunk_begin_ += chunk_length_;
      ++chunk_index_;
      if (chunk_index_ >= chunked_array_->num_chunks()) {
        return Status::Invalid("Adjacency row ", row,
                               " is out of range in chunked array");
      }
      current_chunk_ = std::static_pointer_cast<arrow::Int64Array>(
          chunked_array_->chunk(chunk_index_));
      chunk_length_ = current_chunk_->length();
    }
    return Status::OK();
  }

  Result<IdType> ValueAt(IdType row) {
    GAR_RETURN_NOT_OK(AdvanceTo(row));
    return current_chunk_->Value(row - chunk_begin_);
  }

  Result<const int64_t*> RawValuesAt(IdType row, IdType* available) {
    GAR_RETURN_NOT_OK(AdvanceTo(row));
    const IdType local_row = row - chunk_begin_;
    *available = chunk_length_ - local_row;
    return current_chunk_->raw_values() + local_row;
  }

 private:
  void Reset() {
    chunk_index_ = 0;
    chunk_begin_ = 0;
    chunk_length_ = 0;
    current_chunk_.reset();
    if (chunked_array_ != nullptr && chunked_array_->num_chunks() > 0) {
      current_chunk_ = std::static_pointer_cast<arrow::Int64Array>(
          chunked_array_->chunk(0));
      chunk_length_ = current_chunk_->length();
    }
  }

  std::shared_ptr<arrow::ChunkedArray> chunked_array_;
  int chunk_index_ = 0;
  IdType chunk_begin_ = 0;
  IdType chunk_length_ = 0;
  std::shared_ptr<arrow::Int64Array> current_chunk_;
};

ChunkReadKey MakeOffsetChunkKey(const std::shared_ptr<GraphInfo>& graph_info,
                                const std::string& vertex_type,
                                const std::shared_ptr<EdgeInfo>& edge_info,
                                AdjListType adj_list_type,
                                IdType vertex_chunk_id) {
  ChunkReadKey key;
  key.kind = ChunkReadKind::kEdgeOffset;
  key.graph_prefix = graph_info->GetPrefix();
  key.vertex_type = vertex_type;
  key.edge_type = edge_info->GetEdgeType();
  key.dst_type = vertex_type;
  key.adj_list_type = adj_list_type;
  key.file_type = edge_info->GetAdjacentList(adj_list_type)->GetFileType();
  key.vertex_chunk_id = vertex_chunk_id;
  return key;
}

ChunkReadKey MakeAdjChunkKey(const std::shared_ptr<GraphInfo>& graph_info,
                             const std::string& vertex_type,
                             const std::shared_ptr<EdgeInfo>& edge_info,
                             AdjListType adj_list_type, IdType vertex_chunk_id,
                             IdType chunk_id) {
  ChunkReadKey key = MakeOffsetChunkKey(graph_info, vertex_type, edge_info,
                                        adj_list_type, vertex_chunk_id);
  key.kind = ChunkReadKind::kEdgeAdjList;
  key.chunk_id = chunk_id;
  return key;
}

}  // namespace

struct SamplingBatchState {
  using BatchResult = Result<SamplingResult>;
  using TablePtr = std::shared_ptr<arrow::Table>;

  struct SourceChunkGroup {
    IdType vertex_chunk_id = 0;
    std::vector<IdType> src_nodes;
  };

  struct AdjChunkSpan {
    ChunkReadKey key;
    IdType row_begin = 0;
    IdType row_end = 0;
    std::vector<IdType> picked_rows;
  };

  struct SourcePlan {
    IdType src_node = 0;
    bool needs_shuffle = false;
    std::vector<AdjChunkSpan> spans;
    std::vector<IdType> neighbors;
  };

  struct HopState {
    size_t hop_index = 0;
    int max_neighbors = 0;
    std::vector<IdType> sorted_frontier;
    std::vector<SourceChunkGroup> source_groups;
    std::vector<TablePtr> offset_tables;
    size_t pending_offset_groups = 0;
    std::vector<SourcePlan> source_plans;
    size_t pending_adj_chunks = 0;
  };

  uint64_t id = 0;
  std::shared_ptr<GraphInfo> graph_info;
  std::string vertex_type;
  std::string edge_type;
  std::shared_ptr<EdgeInfo> edge_info;
  AdjListType adj_list_type = AdjListType::ordered_by_source;
  IdType src_chunk_size = 0;
  IdType edge_chunk_size = 0;
  std::vector<int> fanout;
  std::vector<IdType> seed_nodes;
  std::unordered_set<IdType> all_sampled_nodes_set;
  std::vector<IdType> sampled_nodes;
  std::vector<std::pair<IdType, IdType>> edge_list;
  std::vector<IdType> current_frontier;
  std::vector<IdType> num_sampled_nodes_per_hop;
  std::vector<IdType> num_sampled_edges_per_hop;
  size_t current_hop = 0;
  HopState hop_state;
  std::mt19937 gen;
  Clock::time_point submitted_at = Clock::now();
  Clock::time_point ready_at = Clock::now();
  std::promise<BatchResult> promise;
  std::shared_future<BatchResult> future = promise.get_future().share();

  mutable std::mutex mutex_;
  bool promise_set = false;
};

struct EdgeSamplingPipelineCoordinator::Impl {
  using TablePtr = std::shared_ptr<arrow::Table>;
  using BatchResult = Result<SamplingResult>;
  using TableResult = Result<TablePtr>;

  enum class SubscriptionKind { kOffsetGroup, kAdjChunk };

  struct Subscription {
    std::shared_ptr<SamplingBatchState> batch;
    SubscriptionKind kind = SubscriptionKind::kOffsetGroup;
    size_t index = 0;
    ChunkReadKey key;
  };

  struct ActiveChunk {
    ChunkReadKey key;
    OrderedChunkCursor::OrderKey order_key;
    std::shared_ptr<GraphInfo> graph_info;
    std::string vertex_type;
    std::shared_ptr<EdgeInfo> edge_info;
    AdjListType adj_list_type = AdjListType::ordered_by_source;
    IdType vertex_chunk_id = 0;
    IdType chunk_id = 0;
    TablePtr table;
    Status error = Status::UnknownError("chunk not loaded");
    bool ready = false;
    bool failed = false;
    size_t in_flight_processors = 0;
    std::vector<Subscription> waiting_subscribers;
  };

  struct ProcessorTask {
    std::shared_ptr<ActiveChunk> active_chunk;
    std::function<void()> fn;
    Clock::time_point enqueued_at = Clock::now();
  };

  Impl(std::shared_ptr<ChunkReadManager> chunk_manager,
       EdgeSamplingPipelineOptions options)
      : chunk_manager_(std::move(chunk_manager)),
        options_(NormalizeOptions(options)) {
    read_cursor_ = std::make_unique<OrderedChunkCursor>(
        options_.num_readers, 0, false);
    reader_threads_.reserve(options_.num_readers);
    for (size_t i = 0; i < options_.num_readers; ++i) {
      reader_threads_.emplace_back([this]() { ReaderLoop(); });
    }
    processor_threads_.reserve(options_.num_processors);
    for (size_t i = 0; i < options_.num_processors; ++i) {
      processor_threads_.emplace_back([this]() { ProcessorLoop(); });
    }
  }

  ~Impl() { Shutdown(); }

  Result<std::shared_ptr<SamplingBatchHandle>> SubmitSeedBatch(
      const std::shared_ptr<GraphInfo>& graph_info,
      const std::string& vertex_type, const std::string& edge_type,
      const std::vector<IdType>& seed_nodes, const std::vector<int>& fanout,
      uint64_t seed) {
    if (chunk_manager_ == nullptr) {
      return Status::Invalid("Edge pipeline requires a chunk manager");
    }
    if (graph_info == nullptr) {
      return Status::Invalid("GraphInfo cannot be null");
    }
    if (fanout.empty()) {
      return Status::Invalid("Fanout cannot be empty");
    }
    for (size_t hop = 0; hop < fanout.size(); ++hop) {
      if (fanout[hop] < 0) {
        return Status::Invalid("Fanout at hop ", hop, " cannot be negative");
      }
    }

    auto edge_info = graph_info->GetEdgeInfo(vertex_type, edge_type, vertex_type);
    if (!edge_info) {
      return Status::Invalid("Edge type '", edge_type,
                             "' not found for vertex type '", vertex_type, "'");
    }
    if (!edge_info->HasAdjacentListType(AdjListType::ordered_by_source)) {
      return Status::Invalid(
          "CSR layout (ordered_by_source) not available for edge type '",
          edge_type, "'");
    }

    auto batch = std::make_shared<SamplingBatchState>();
    batch->id = next_batch_id_.fetch_add(1, std::memory_order_relaxed);
    batch->graph_info = graph_info;
    batch->vertex_type = vertex_type;
    batch->edge_type = edge_type;
    batch->edge_info = edge_info;
    batch->fanout = fanout;
    batch->seed_nodes = seed_nodes;
    batch->src_chunk_size = edge_info->GetSrcChunkSize();
    batch->edge_chunk_size = edge_info->GetChunkSize();
    batch->gen.seed(seed);
    batch->submitted_at = Clock::now();

    for (IdType seed_node : seed_nodes) {
      if (batch->all_sampled_nodes_set.insert(seed_node).second) {
        batch->sampled_nodes.push_back(seed_node);
      }
    }
    batch->current_frontier = seed_nodes;
    batch->num_sampled_nodes_per_hop.push_back(batch->sampled_nodes.size());
    submitted_batches_.fetch_add(1, std::memory_order_relaxed);

    {
      std::unique_lock<std::mutex> lock(mutex_);
      active_batches_cv_.wait(lock, [&]() {
        return shutdown_ ||
               active_batches_.size() < options_.max_active_batches;
      });
      if (shutdown_) {
        return Status::Invalid("Edge pipeline is shut down");
      }
      active_batches_.emplace(batch->id, batch);
      AtomicMax(&pending_batches_peak_,
                static_cast<uint64_t>(active_batches_.size()));
    }

    if (seed_nodes.empty()) {
      SamplingResult empty;
      empty.num_sampled_nodes_per_hop.assign(fanout.size() + 1, 0);
      empty.num_sampled_edges_per_hop.assign(fanout.size(), 0);
      FinishBatch(batch, empty);
      return std::make_shared<SamplingBatchHandle>(batch);
    }

    auto start_status = StartOffsetPhase(batch);
    if (!start_status.ok()) {
      FinishBatch(batch, start_status);
    }
    return std::make_shared<SamplingBatchHandle>(batch);
  }

  EdgeSamplingPipelineStats Stats() const {
    EdgeSamplingPipelineStats stats;
    stats.submitted_batches =
        submitted_batches_.load(std::memory_order_relaxed);
    stats.completed_batches =
        completed_batches_.load(std::memory_order_relaxed);
    stats.pending_batches_peak =
        pending_batches_peak_.load(std::memory_order_relaxed);
    stats.active_offset_chunk_keys_peak =
        active_offset_chunk_keys_peak_.load(std::memory_order_relaxed);
    stats.active_adj_chunk_keys_peak =
        active_adj_chunk_keys_peak_.load(std::memory_order_relaxed);
    stats.offset_chunk_subscriptions =
        offset_chunk_subscriptions_.load(std::memory_order_relaxed);
    stats.offset_chunk_reads =
        offset_chunk_reads_.load(std::memory_order_relaxed);
    stats.offset_chunk_reuses =
        offset_chunk_reuses_.load(std::memory_order_relaxed);
    stats.adj_chunk_subscriptions =
        adj_chunk_subscriptions_.load(std::memory_order_relaxed);
    stats.adj_chunk_reads = adj_chunk_reads_.load(std::memory_order_relaxed);
    stats.adj_chunk_reuses =
        adj_chunk_reuses_.load(std::memory_order_relaxed);
    stats.processor_tasks =
        processor_tasks_.load(std::memory_order_relaxed);
    stats.processor_wait_ms_sum =
        processor_wait_ms_sum_.load(std::memory_order_relaxed);
    stats.processor_service_ms_sum =
        processor_service_ms_sum_.load(std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stats.active_batches_current = active_batches_.size();
      stats.active_offset_chunk_keys_current = active_offset_chunk_keys_current_;
      stats.active_adj_chunk_keys_current = active_adj_chunk_keys_current_;
      stats.read_queue_current = read_queue_.size();
      stats.processor_queue_current = processor_queue_.size();
    }
    return stats;
  }

  void Shutdown() {
    std::vector<std::shared_ptr<SamplingBatchState>> batches_to_fail;
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
      processor_queue_.clear();
      active_chunks_.clear();
      active_offset_chunk_keys_current_ = 0;
      active_adj_chunk_keys_current_ = 0;
    }

    active_batches_cv_.notify_all();
    read_cv_.notify_all();
    processor_cv_.notify_all();
    processor_space_cv_.notify_all();

    for (const auto& batch : batches_to_fail) {
      FinishBatch(batch, Status::Invalid("Edge pipeline is shut down"));
    }

    if (read_cursor_ != nullptr) {
      read_cursor_->Shutdown();
    }
    for (auto& thread : reader_threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    reader_threads_.clear();

    for (auto& thread : processor_threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    processor_threads_.clear();
  }

 private:
  Status StartOffsetPhase(const std::shared_ptr<SamplingBatchState>& batch) {
    bool finalize = false;
    size_t hop_index = 0;
    int max_neighbors = 0;
    IdType src_chunk_size = 0;
    std::vector<IdType> sorted_frontier;
    std::vector<SamplingBatchState::SourceChunkGroup> groups;

    {
      std::lock_guard<std::mutex> lock(batch->mutex_);
      if (batch->promise_set) {
        return Status::OK();
      }
      if (batch->current_hop >= batch->fanout.size()) {
        finalize = true;
      } else if (batch->current_frontier.empty()) {
        while (batch->num_sampled_nodes_per_hop.size() <
               batch->fanout.size() + 1) {
          batch->num_sampled_nodes_per_hop.push_back(0);
        }
        while (batch->num_sampled_edges_per_hop.size() < batch->fanout.size()) {
          batch->num_sampled_edges_per_hop.push_back(0);
        }
        finalize = true;
      }
      if (finalize) {
        // finalized after releasing the batch lock
      } else {
        hop_index = batch->current_hop;
        max_neighbors = batch->fanout[hop_index];
        src_chunk_size = batch->src_chunk_size;
        sorted_frontier = batch->current_frontier;
        std::sort(sorted_frontier.begin(), sorted_frontier.end());
        for (size_t fi = 0; fi < sorted_frontier.size();) {
          SamplingBatchState::SourceChunkGroup group;
          group.vertex_chunk_id = sorted_frontier[fi] / src_chunk_size;
          size_t group_end = fi + 1;
          while (group_end < sorted_frontier.size() &&
                 sorted_frontier[group_end] / src_chunk_size ==
                     group.vertex_chunk_id) {
            ++group_end;
          }
          group.src_nodes.assign(sorted_frontier.begin() + fi,
                                 sorted_frontier.begin() + group_end);
          groups.push_back(std::move(group));
          fi = group_end;
        }

        batch->hop_state = SamplingBatchState::HopState{};
        batch->hop_state.hop_index = hop_index;
        batch->hop_state.max_neighbors = max_neighbors;
        batch->hop_state.sorted_frontier = sorted_frontier;
        batch->hop_state.source_groups = groups;
        batch->hop_state.offset_tables.resize(groups.size());
        batch->hop_state.pending_offset_groups = groups.size();
      }
    }

    if (finalize) {
      return FinalizeCompletedBatch(batch);
    }

    if (groups.empty()) {
      return EnqueueProcessorTask(nullptr, [this, batch]() {
        auto status = FinalizeHop(batch);
        if (!status.ok()) {
          FinishBatch(batch, status);
        }
      });
    }

    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
      auto status = AddOffsetSubscription(batch, group_index,
                                          groups[group_index].vertex_chunk_id);
      if (!status.ok()) {
        return status;
      }
    }
    return Status::OK();
  }

  Status AdvanceFromOffsets(const std::shared_ptr<SamplingBatchState>& batch) {
    std::map<std::pair<IdType, IdType>, ChunkReadKey> unique_adj_keys;

    {
      std::lock_guard<std::mutex> lock(batch->mutex_);
      if (batch->promise_set) {
        return Status::OK();
      }

      auto& hop = batch->hop_state;
      hop.source_plans.clear();
      hop.pending_adj_chunks = 0;

      for (size_t group_index = 0; group_index < hop.source_groups.size();
           ++group_index) {
        const auto& group = hop.source_groups[group_index];
        const auto& offset_table = hop.offset_tables[group_index];
        if (offset_table == nullptr || offset_table->num_columns() == 0) {
          return Status::Invalid("Offset chunk has no columns");
        }
        auto offset_column = offset_table->column(0);
        Int64ChunkedArrayCursor offset_cursor(offset_column);
        const IdType offset_len = offset_column->length();

        for (IdType src_node : group.src_nodes) {
          const IdType local_idx = src_node - group.vertex_chunk_id * batch->src_chunk_size;
          if (local_idx + 1 >= offset_len) {
            continue;
          }

          auto begin_result = offset_cursor.ValueAt(local_idx);
          if (begin_result.has_error()) {
            return begin_result.status();
          }
          const IdType begin_off = begin_result.value();
          auto end_result = offset_cursor.ValueAt(local_idx + 1);
          if (end_result.has_error()) {
            return end_result.status();
          }
          const IdType end_off = end_result.value();
          if (begin_off >= end_off || hop.max_neighbors <= 0) {
            if (begin_off < end_off) {
              hop.source_plans.push_back(
                  SamplingBatchState::SourcePlan{src_node, false, {}});
            }
            continue;
          }

          const IdType total_edges = end_off - begin_off;
          SamplingBatchState::SourcePlan plan;
          plan.src_node = src_node;
          if (hop.max_neighbors > 0) {
            plan.neighbors.reserve(
                static_cast<size_t>(
                    std::min<IdType>(total_edges, hop.max_neighbors)));
          }

          if (total_edges <= static_cast<IdType>(hop.max_neighbors)) {
            IdType cur_off = begin_off;
            while (cur_off < end_off) {
              const IdType chunk_id = cur_off / batch->edge_chunk_size;
              const IdType chunk_start = chunk_id * batch->edge_chunk_size;
              const IdType chunk_end =
                  std::min(end_off, chunk_start + batch->edge_chunk_size);
              auto key = MakeAdjChunkKey(batch->graph_info, batch->vertex_type,
                                         batch->edge_info,
                                         batch->adj_list_type,
                                         group.vertex_chunk_id, chunk_id);
              plan.spans.push_back({key, cur_off - chunk_start,
                                    chunk_end - chunk_start, {}});
              unique_adj_keys.emplace(std::make_pair(group.vertex_chunk_id, chunk_id),
                                      key);
              cur_off = chunk_end;
            }
          } else {
            std::vector<IdType> local_indices;
            SampleUniformLocalIndices(total_edges,
                                      static_cast<IdType>(hop.max_neighbors),
                                      &batch->gen, &local_indices);
            plan.needs_shuffle = true;
            size_t pick_i = 0;
            while (pick_i < local_indices.size()) {
              const IdType abs_off = begin_off + local_indices[pick_i];
              const IdType chunk_id = abs_off / batch->edge_chunk_size;
              const IdType chunk_start = chunk_id * batch->edge_chunk_size;
              auto key = MakeAdjChunkKey(batch->graph_info, batch->vertex_type,
                                         batch->edge_info,
                                         batch->adj_list_type,
                                         group.vertex_chunk_id, chunk_id);
              SamplingBatchState::AdjChunkSpan span;
              span.key = key;
              while (pick_i < local_indices.size()) {
                const IdType abs_off2 = begin_off + local_indices[pick_i];
                if (abs_off2 / batch->edge_chunk_size != chunk_id) {
                  break;
                }
                span.picked_rows.push_back(abs_off2 - chunk_start);
                ++pick_i;
              }
              plan.spans.push_back(std::move(span));
              unique_adj_keys.emplace(std::make_pair(group.vertex_chunk_id, chunk_id),
                                      key);
            }
          }

          hop.source_plans.push_back(std::move(plan));
        }
      }

      hop.pending_adj_chunks = unique_adj_keys.size();
      hop.offset_tables.clear();
      hop.offset_tables.shrink_to_fit();
      hop.source_groups.clear();
      hop.source_groups.shrink_to_fit();
      hop.sorted_frontier.clear();
      hop.sorted_frontier.shrink_to_fit();
    }

    if (unique_adj_keys.empty()) {
      return EnqueueProcessorTask(nullptr, [this, batch]() {
        auto status = FinalizeHop(batch);
        if (!status.ok()) {
          FinishBatch(batch, status);
        }
      });
    }

    for (const auto& [_, key] : unique_adj_keys) {
      auto status = AddAdjSubscription(batch, key);
      if (!status.ok()) {
        return status;
      }
    }
    return Status::OK();
  }

  Status FinalizeHop(const std::shared_ptr<SamplingBatchState>& batch) {
    {
      std::lock_guard<std::mutex> lock(batch->mutex_);
      if (batch->promise_set) {
        return Status::OK();
      }

      auto& hop = batch->hop_state;
      std::vector<IdType> next_frontier;
      IdType hop_num_sampled_nodes = 0;
      IdType hop_num_sampled_edges = 0;

      for (auto& plan : hop.source_plans) {
        if (plan.needs_shuffle) {
          std::shuffle(plan.neighbors.begin(), plan.neighbors.end(), batch->gen);
        }

        for (IdType dst_node : plan.neighbors) {
          batch->edge_list.emplace_back(plan.src_node, dst_node);
          ++hop_num_sampled_edges;
          if (batch->all_sampled_nodes_set.insert(dst_node).second) {
            batch->sampled_nodes.push_back(dst_node);
            next_frontier.push_back(dst_node);
            ++hop_num_sampled_nodes;
          }
        }
      }

      batch->num_sampled_nodes_per_hop.push_back(hop_num_sampled_nodes);
      batch->num_sampled_edges_per_hop.push_back(hop_num_sampled_edges);
      batch->current_frontier = std::move(next_frontier);
      batch->current_hop += 1;
      batch->hop_state = SamplingBatchState::HopState{};
    }

    return StartOffsetPhase(batch);
  }

  Status FinalizeCompletedBatch(const std::shared_ptr<SamplingBatchState>& batch) {
    SamplingResult result;
    {
      std::lock_guard<std::mutex> lock(batch->mutex_);
      result.sampled_nodes = batch->sampled_nodes;
      result.num_sampled_nodes_per_hop = batch->num_sampled_nodes_per_hop;
      result.num_sampled_edges_per_hop = batch->num_sampled_edges_per_hop;
      std::unordered_map<IdType, IdType> node_to_index;
      for (size_t i = 0; i < result.sampled_nodes.size(); ++i) {
        node_to_index[result.sampled_nodes[i]] = i;
      }
      result.src_indices.reserve(batch->edge_list.size());
      result.dst_indices.reserve(batch->edge_list.size());
      for (const auto& [src, dst] : batch->edge_list) {
        result.src_indices.push_back(node_to_index[src]);
        result.dst_indices.push_back(node_to_index[dst]);
      }
    }
    FinishBatch(batch, result);
    return Status::OK();
  }

  Status AddOffsetSubscription(const std::shared_ptr<SamplingBatchState>& batch,
                               size_t group_index, IdType vertex_chunk_id) {
    Subscription subscription;
    subscription.batch = batch;
    subscription.kind = SubscriptionKind::kOffsetGroup;
    subscription.index = group_index;

    auto key = MakeOffsetChunkKey(batch->graph_info, batch->vertex_type,
                                  batch->edge_info, batch->adj_list_type,
                                  vertex_chunk_id);
    auto order_key = OrderedChunkCursor::OrderKey{vertex_chunk_id, 0, 0};
    return AddSubscription(std::move(subscription), key, order_key,
                           batch->graph_info, batch->vertex_type,
                           batch->edge_info, batch->adj_list_type,
                           vertex_chunk_id, 0);
  }

  Status AddAdjSubscription(const std::shared_ptr<SamplingBatchState>& batch,
                            const ChunkReadKey& key) {
    Subscription subscription;
    subscription.batch = batch;
    subscription.kind = SubscriptionKind::kAdjChunk;
    subscription.key = key;
    auto order_key =
        OrderedChunkCursor::OrderKey{key.vertex_chunk_id, 1, key.chunk_id};
    return AddSubscription(std::move(subscription), key, order_key,
                           batch->graph_info, batch->vertex_type,
                           batch->edge_info, batch->adj_list_type,
                           key.vertex_chunk_id, key.chunk_id);
  }

  Status AddSubscription(Subscription subscription, const ChunkReadKey& key,
                         const OrderedChunkCursor::OrderKey& order_key,
                         const std::shared_ptr<GraphInfo>& graph_info,
                         const std::string& vertex_type,
                         const std::shared_ptr<EdgeInfo>& edge_info,
                         AdjListType adj_list_type, IdType vertex_chunk_id,
                         IdType chunk_id) {
    std::shared_ptr<ActiveChunk> active_chunk;
    TablePtr ready_table;
    std::unique_ptr<Status> chunk_error;
    bool enqueue_read = false;
    bool enqueue_processor = false;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (shutdown_) {
        return Status::Invalid("Edge pipeline is shut down");
      }

      auto* subscriptions = subscription.kind == SubscriptionKind::kOffsetGroup
                                ? &offset_chunk_subscriptions_
                                : &adj_chunk_subscriptions_;
      auto* reuses = subscription.kind == SubscriptionKind::kOffsetGroup
                         ? &offset_chunk_reuses_
                         : &adj_chunk_reuses_;
      subscriptions->fetch_add(1, std::memory_order_relaxed);

      auto it = active_chunks_.find(key);
      if (it == active_chunks_.end()) {
        active_chunk = std::make_shared<ActiveChunk>();
        active_chunk->key = key;
        active_chunk->order_key = order_key;
        active_chunk->graph_info = graph_info;
        active_chunk->vertex_type = vertex_type;
        active_chunk->edge_info = edge_info;
        active_chunk->adj_list_type = adj_list_type;
        active_chunk->vertex_chunk_id = vertex_chunk_id;
        active_chunk->chunk_id = chunk_id;
        active_chunk->waiting_subscribers.push_back(std::move(subscription));
        active_chunks_.emplace(key, active_chunk);
        read_queue_.push_back(active_chunk);
        if (key.kind == ChunkReadKind::kEdgeOffset) {
          active_offset_chunk_keys_current_ += 1;
          AtomicMax(&active_offset_chunk_keys_peak_,
                    static_cast<uint64_t>(active_offset_chunk_keys_current_));
        } else {
          active_adj_chunk_keys_current_ += 1;
          AtomicMax(&active_adj_chunk_keys_peak_,
                    static_cast<uint64_t>(active_adj_chunk_keys_current_));
        }
        enqueue_read = true;
      } else {
        active_chunk = it->second;
        reuses->fetch_add(1, std::memory_order_relaxed);
        if (active_chunk->failed) {
          chunk_error = std::make_unique<Status>(active_chunk->error);
        } else if (active_chunk->ready) {
          active_chunk->in_flight_processors += 1;
          ready_table = active_chunk->table;
          enqueue_processor = true;
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
    if (!enqueue_processor) {
      return Status::OK();
    }
    return EnqueueProcessorForSubscription(active_chunk, std::move(subscription),
                                           ready_table);
  }

  Status EnqueueProcessorForSubscription(
      const std::shared_ptr<ActiveChunk>& active_chunk, Subscription subscription,
      const TablePtr& table) {
    if (subscription.kind == SubscriptionKind::kOffsetGroup) {
      auto batch = subscription.batch;
      const auto group_index = subscription.index;
      return EnqueueProcessorTask(active_chunk, [this, batch, group_index, table]() {
        auto status = OnOffsetReady(batch, group_index, table);
        if (!status.ok()) {
          FinishBatch(batch, status);
        }
      });
    }

    auto batch = subscription.batch;
    auto key = subscription.key;
    return EnqueueProcessorTask(active_chunk, [this, batch, key, table]() {
      auto status = OnAdjReady(batch, key, table);
      if (!status.ok()) {
        FinishBatch(batch, status);
      }
    });
  }

  Status EnqueueProcessorTask(const std::shared_ptr<ActiveChunk>& active_chunk,
                              std::function<void()> fn) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      processor_space_cv_.wait(lock, [&]() {
        return shutdown_ ||
               processor_queue_.size() < options_.max_queued_processor_tasks;
      });
      if (shutdown_) {
        if (active_chunk != nullptr && active_chunk->in_flight_processors > 0) {
          active_chunk->in_flight_processors -= 1;
        }
        return Status::Invalid("Edge pipeline is shut down");
      }
      processor_queue_.push_back(
          ProcessorTask{active_chunk, std::move(fn), Clock::now()});
      processor_tasks_.fetch_add(1, std::memory_order_relaxed);
    }
    processor_cv_.notify_one();
    return Status::OK();
  }

  Status OnOffsetReady(const std::shared_ptr<SamplingBatchState>& batch,
                       size_t group_index, const TablePtr& table) {
    bool advance = false;
    {
      std::lock_guard<std::mutex> lock(batch->mutex_);
      if (batch->promise_set) {
        return Status::OK();
      }
      if (group_index >= batch->hop_state.offset_tables.size()) {
        return Status::Invalid("Offset subscription index out of range");
      }
      batch->hop_state.offset_tables[group_index] = table;
      if (batch->hop_state.pending_offset_groups == 0) {
        return Status::Invalid("Offset pending counter underflow");
      }
      batch->hop_state.pending_offset_groups -= 1;
      advance = batch->hop_state.pending_offset_groups == 0;
    }

    if (!advance) {
      return Status::OK();
    }
    return EnqueueProcessorTask(nullptr, [this, batch]() {
      auto status = AdvanceFromOffsets(batch);
      if (!status.ok()) {
        FinishBatch(batch, status);
      }
    });
  }

  Status OnAdjReady(const std::shared_ptr<SamplingBatchState>& batch,
                    const ChunkReadKey& key, const TablePtr& table) {
    if (table == nullptr) {
      return Status::Invalid("Adjacency chunk is null");
    }
    if (table->num_columns() < 2) {
      return Status::Invalid("Adjacency chunk has fewer than 2 columns");
    }

    Int64ChunkedArrayCursor cursor(table->column(1));
    bool advance = false;
    {
      std::lock_guard<std::mutex> lock(batch->mutex_);
      if (batch->promise_set) {
        return Status::OK();
      }
      for (auto& plan : batch->hop_state.source_plans) {
        for (const auto& span : plan.spans) {
          if (!(span.key == key)) {
            continue;
          }
          if (span.picked_rows.empty()) {
            IdType row = span.row_begin;
            while (row < span.row_end) {
              IdType available = 0;
              auto raw_result = cursor.RawValuesAt(row, &available);
              if (raw_result.has_error()) {
                return raw_result.status();
              }
              auto raw = raw_result.value();
              const IdType batch_rows = std::min(span.row_end - row, available);
              plan.neighbors.insert(plan.neighbors.end(), raw, raw + batch_rows);
              row += batch_rows;
            }
          } else {
            for (IdType row : span.picked_rows) {
              auto dst_result = cursor.ValueAt(row);
              if (dst_result.has_error()) {
                return dst_result.status();
              }
              plan.neighbors.push_back(dst_result.value());
            }
          }
        }
      }
      if (batch->hop_state.pending_adj_chunks == 0) {
        return Status::Invalid("Adjacency pending counter underflow");
      }
      batch->hop_state.pending_adj_chunks -= 1;
      advance = batch->hop_state.pending_adj_chunks == 0;
    }

    if (!advance) {
      return Status::OK();
    }
    return EnqueueProcessorTask(nullptr, [this, batch]() {
      auto status = FinalizeHop(batch);
      if (!status.ok()) {
        FinishBatch(batch, status);
      }
    });
  }

  void ReaderLoop() {
    while (true) {
      std::shared_ptr<ActiveChunk> active_chunk;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        read_cv_.wait(lock, [&]() { return shutdown_ || !read_queue_.empty(); });
        if (shutdown_ && read_queue_.empty()) {
          return;
        }
        active_chunk = read_queue_.front();
        read_queue_.pop_front();
      }

      if (active_chunk == nullptr) {
        continue;
      }

      auto direct_loader = [this, active_chunk]() -> TableResult {
        if (active_chunk->key.kind == ChunkReadKind::kEdgeOffset) {
          return chunk_manager_->GetEdgeOffsetChunk(
              active_chunk->graph_info, active_chunk->vertex_type,
              active_chunk->edge_info->GetEdgeType(), active_chunk->vertex_type,
              active_chunk->adj_list_type, active_chunk->vertex_chunk_id);
        }
        return chunk_manager_->GetEdgeAdjListChunk(
            active_chunk->graph_info, active_chunk->vertex_type,
            active_chunk->edge_info->GetEdgeType(), active_chunk->vertex_type,
            active_chunk->adj_list_type, active_chunk->vertex_chunk_id,
            active_chunk->chunk_id);
      };
      auto table_result = read_cursor_->LoadChunk(active_chunk->key,
                                                  active_chunk->order_key,
                                                  direct_loader);

      if (active_chunk->key.kind == ChunkReadKind::kEdgeOffset) {
        offset_chunk_reads_.fetch_add(1, std::memory_order_relaxed);
      } else {
        adj_chunk_reads_.fetch_add(1, std::memory_order_relaxed);
      }

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
          RemoveActiveChunkLocked(it);
        } else if (table_result.has_error()) {
          current_chunk->failed = true;
          current_chunk->error = table_result.status();
          RemoveActiveChunkLocked(it);
        } else {
          current_chunk->ready = true;
          current_chunk->table = table_result.value();
          current_chunk->in_flight_processors += subscriptions.size();
          if (subscriptions.empty() &&
              current_chunk->in_flight_processors == 0) {
            RemoveActiveChunkLocked(it);
          }
        }
      }

      if (shutdown_) {
        for (const auto& subscription : subscriptions) {
          FinishBatch(subscription.batch,
                      Status::Invalid("Edge pipeline is shut down"));
        }
        continue;
      }

      if (table_result.has_error()) {
        for (const auto& subscription : subscriptions) {
          FinishBatch(subscription.batch, table_result.status());
        }
        continue;
      }

      for (auto& subscription : subscriptions) {
        auto batch = subscription.batch;
        auto status = EnqueueProcessorForSubscription(
            current_chunk, std::move(subscription), table_result.value());
        if (!status.ok()) {
          FinishBatch(batch, status);
        }
      }
    }
  }

  void ProcessorLoop() {
    while (true) {
      ProcessorTask task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        processor_cv_.wait(lock,
                           [&]() { return shutdown_ || !processor_queue_.empty(); });
        if (shutdown_ && processor_queue_.empty()) {
          return;
        }
        task = std::move(processor_queue_.front());
        processor_queue_.pop_front();
      }
      processor_space_cv_.notify_all();

      const auto service_start = Clock::now();
      const auto wait_ms = MillisecondsBetween(task.enqueued_at, service_start);
      processor_wait_ms_sum_.fetch_add(wait_ms, std::memory_order_relaxed);

      task.fn();

      const auto service_ms = MillisecondsBetween(service_start, Clock::now());
      processor_service_ms_sum_.fetch_add(service_ms,
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
    if (it->second->in_flight_processors > 0) {
      it->second->in_flight_processors -= 1;
    }
    if (it->second->ready && it->second->waiting_subscribers.empty() &&
        it->second->in_flight_processors == 0) {
      RemoveActiveChunkLocked(it);
    }
  }

  void RemoveActiveChunkLocked(
      const std::unordered_map<ChunkReadKey, std::shared_ptr<ActiveChunk>,
                               ChunkReadKeyHash>::iterator& it) {
    if (it->first.kind == ChunkReadKind::kEdgeOffset) {
      if (active_offset_chunk_keys_current_ > 0) {
        active_offset_chunk_keys_current_ -= 1;
      }
    } else if (active_adj_chunk_keys_current_ > 0) {
      active_adj_chunk_keys_current_ -= 1;
    }
    active_chunks_.erase(it);
  }

  void FinishBatch(const std::shared_ptr<SamplingBatchState>& batch,
                   const BatchResult& result) {
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
    completed_batches_.fetch_add(1, std::memory_order_relaxed);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      active_batches_.erase(batch->id);
    }
    active_batches_cv_.notify_all();
  }

  std::shared_ptr<ChunkReadManager> chunk_manager_;
  EdgeSamplingPipelineOptions options_;
  std::unique_ptr<OrderedChunkCursor> read_cursor_;

  mutable std::mutex mutex_;
  std::condition_variable active_batches_cv_;
  std::condition_variable read_cv_;
  std::condition_variable processor_cv_;
  std::condition_variable processor_space_cv_;
  bool shutdown_ = false;

  std::unordered_map<uint64_t, std::shared_ptr<SamplingBatchState>>
      active_batches_;
  std::unordered_map<ChunkReadKey, std::shared_ptr<ActiveChunk>,
                     ChunkReadKeyHash>
      active_chunks_;
  std::deque<std::shared_ptr<ActiveChunk>> read_queue_;
  std::deque<ProcessorTask> processor_queue_;
  std::vector<std::thread> reader_threads_;
  std::vector<std::thread> processor_threads_;
  size_t active_offset_chunk_keys_current_ = 0;
  size_t active_adj_chunk_keys_current_ = 0;

  std::atomic<uint64_t> next_batch_id_{0};
  std::atomic<uint64_t> submitted_batches_{0};
  std::atomic<uint64_t> completed_batches_{0};
  std::atomic<uint64_t> pending_batches_peak_{0};
  std::atomic<uint64_t> active_offset_chunk_keys_peak_{0};
  std::atomic<uint64_t> active_adj_chunk_keys_peak_{0};
  std::atomic<uint64_t> offset_chunk_subscriptions_{0};
  std::atomic<uint64_t> offset_chunk_reads_{0};
  std::atomic<uint64_t> offset_chunk_reuses_{0};
  std::atomic<uint64_t> adj_chunk_subscriptions_{0};
  std::atomic<uint64_t> adj_chunk_reads_{0};
  std::atomic<uint64_t> adj_chunk_reuses_{0};
  std::atomic<uint64_t> processor_tasks_{0};
  std::atomic<uint64_t> processor_wait_ms_sum_{0};
  std::atomic<uint64_t> processor_service_ms_sum_{0};
};

SamplingBatchHandle::SamplingBatchHandle() = default;

SamplingBatchHandle::SamplingBatchHandle(std::shared_ptr<SamplingBatchState> state)
    : state_(std::move(state)) {}

SamplingBatchHandle::~SamplingBatchHandle() = default;

SamplingBatchHandle::SamplingBatchHandle(const SamplingBatchHandle&) = default;

SamplingBatchHandle& SamplingBatchHandle::operator=(
    const SamplingBatchHandle&) = default;

SamplingBatchHandle::SamplingBatchHandle(
    SamplingBatchHandle&&) noexcept = default;

SamplingBatchHandle& SamplingBatchHandle::operator=(
    SamplingBatchHandle&&) noexcept = default;

SamplingBatchHandle::BatchResult SamplingBatchHandle::Wait() const {
  if (state_ == nullptr) {
    return Status::Invalid("Sampling batch handle is empty");
  }
  return state_->future.get();
}

uint64_t SamplingBatchHandle::sampling_ms() const {
  if (state_ == nullptr) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(state_->mutex_);
  if (!state_->promise_set) {
    return 0;
  }
  return MillisecondsBetween(state_->submitted_at, state_->ready_at);
}

bool SamplingBatchHandle::valid() const { return state_ != nullptr; }

EdgeSamplingPipelineCoordinator::EdgeSamplingPipelineCoordinator(
    std::shared_ptr<ChunkReadManager> chunk_manager,
    EdgeSamplingPipelineOptions options)
    : impl_(std::make_shared<Impl>(std::move(chunk_manager),
                                   std::move(options))) {}

EdgeSamplingPipelineCoordinator::~EdgeSamplingPipelineCoordinator() = default;

EdgeSamplingPipelineCoordinator::EdgeSamplingPipelineCoordinator(
    const EdgeSamplingPipelineCoordinator&) = default;

EdgeSamplingPipelineCoordinator& EdgeSamplingPipelineCoordinator::operator=(
    const EdgeSamplingPipelineCoordinator&) = default;

EdgeSamplingPipelineCoordinator::EdgeSamplingPipelineCoordinator(
    EdgeSamplingPipelineCoordinator&&) noexcept = default;

EdgeSamplingPipelineCoordinator& EdgeSamplingPipelineCoordinator::operator=(
    EdgeSamplingPipelineCoordinator&&) noexcept = default;

Result<std::shared_ptr<SamplingBatchHandle>>
EdgeSamplingPipelineCoordinator::SubmitSeedBatch(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::string& vertex_type, const std::string& edge_type,
    const std::vector<IdType>& seed_nodes, const std::vector<int>& fanout,
    uint64_t seed) {
  return impl_->SubmitSeedBatch(graph_info, vertex_type, edge_type, seed_nodes,
                                fanout, seed);
}

EdgeSamplingPipelineStats EdgeSamplingPipelineCoordinator::Stats() const {
  return impl_->Stats();
}

void EdgeSamplingPipelineCoordinator::Shutdown() { impl_->Shutdown(); }

}  // namespace graphar::ml
