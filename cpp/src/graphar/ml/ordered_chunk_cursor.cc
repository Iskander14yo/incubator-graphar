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

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graphar/status.h"

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

uint64_t MillisecondsSince(Clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

}  // namespace

bool OrderedChunkCursor::OrderKey::operator<(const OrderKey& other) const {
  return primary < other.primary ||
         (primary == other.primary &&
          (secondary < other.secondary ||
           (secondary == other.secondary && tertiary < other.tertiary)));
}

struct OrderedChunkCursor::Impl {
  struct TrailEntry {
    ChunkReadKey key;
    TablePtr table;
  };

  struct Task {
    ChunkReadKey key;
    OrderKey order_key;
    TableLoader loader;
    CompletionCallback on_complete;
    bool track_request = false;
    Clock::time_point request_start;
  };

  struct QueueState {
    std::map<OrderKey, std::deque<std::shared_ptr<Task>>> tasks_by_order_key;
    size_t task_count = 0;
    OrderKey next_order_key;
    bool has_next_order_key = false;
  };

  Impl(size_t cursor_count, size_t trail_capacity_chunks,
       bool auto_track_requests)
      : cursor_count_(std::max<size_t>(1, cursor_count)),
        trail_capacity_chunks_(trail_capacity_chunks),
        auto_track_requests_(auto_track_requests),
        queues_(cursor_count_) {
    workers_.reserve(cursor_count_);
    for (size_t i = 0; i < cursor_count_; ++i) {
      workers_.emplace_back([this, i]() { WorkerLoop(i); });
    }
  }

  ~Impl() { Shutdown(); }

  TableResult LoadChunk(const ChunkReadKey& key, OrderKey order_key,
                        const TableLoader& loader) {
    auto promise = std::make_shared<std::promise<TableResult>>();
    auto future = promise->get_future();
    auto status = SubmitChunk(
        key, order_key, loader,
        [promise](TableResult result) { promise->set_value(std::move(result)); });
    if (!status.ok()) {
      return status;
    }
    return future.get();
  }

  Status SubmitChunk(const ChunkReadKey& key, OrderKey order_key,
                     const TableLoader& loader,
                     CompletionCallback on_complete) {
    const auto request_start = Clock::now();
    TablePtr trail_table;
    bool served_from_trail = false;
    bool track_request = auto_track_requests_;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        return Status::Invalid("Chunk cursor is shut down");
      }
      if (track_request) {
        RegisterRequest();
      }
      trail_table = LookupTrailLocked(key);
      if (trail_table != nullptr) {
        trail_hits_.fetch_add(1, std::memory_order_relaxed);
        served_from_trail = true;
      } else {
        trail_misses_.fetch_add(1, std::memory_order_relaxed);
        auto task = std::make_shared<Task>();
        task->key = key;
        task->order_key = order_key;
        task->loader = loader;
        task->on_complete = std::move(on_complete);
        task->track_request = track_request;
        task->request_start = request_start;
        auto& queue = queues_[CursorIndexFor(key)];
        queue.tasks_by_order_key[task->order_key].push_back(task);
        queue.task_count += 1;
      }
    }

    if (served_from_trail) {
      if (track_request) {
        CompleteRequest(request_start, true);
      }
      on_complete(trail_table);
      return Status::OK();
    }

    cv_.notify_all();
    return Status::OK();
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
    stats.chunk_order_wraps =
        chunk_order_wraps_.load(std::memory_order_relaxed);
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
        for (auto& [_, tasks] : queue.tasks_by_order_key) {
          while (!tasks.empty()) {
            pending.push_back(std::move(tasks.front()));
            tasks.pop_front();
          }
        }
        queue.tasks_by_order_key.clear();
        queue.task_count = 0;
      }
    }

    cv_.notify_all();
    for (const auto& task : pending) {
      if (task->track_request) {
        CompleteRequest(task->request_start, false);
      }
      task->on_complete(Status::Invalid("Chunk cursor is shut down"));
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

  std::shared_ptr<Task> PopNextTaskLocked(size_t index, bool* wrapped) {
    auto& queue = queues_[index];
    if (queue.task_count == 0) {
      return nullptr;
    }

    auto it = queue.tasks_by_order_key.begin();
    if (queue.has_next_order_key) {
      it = queue.tasks_by_order_key.lower_bound(queue.next_order_key);
      if (it == queue.tasks_by_order_key.end()) {
        it = queue.tasks_by_order_key.begin();
        *wrapped = true;
      }
    }

    const auto order_key = it->first;
    auto task = std::move(it->second.front());
    it->second.pop_front();
    if (it->second.empty()) {
      queue.tasks_by_order_key.erase(it);
    }
    queue.task_count -= 1;
    queue.next_order_key = order_key;
    queue.has_next_order_key = true;
    return task;
  }

  void WorkerLoop(size_t index) {
    while (true) {
      std::shared_ptr<Task> task;
      bool wrapped = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock,
                 [&]() { return shutdown_ || queues_[index].task_count > 0; });
        if (shutdown_ && queues_[index].task_count == 0) {
          return;
        }
        task = PopNextTaskLocked(index, &wrapped);
      }

      if (wrapped) {
        chunk_order_wraps_.fetch_add(1, std::memory_order_relaxed);
      }

      if (IsShutdown()) {
        if (task->track_request) {
          CompleteRequest(task->request_start, false);
        }
        task->on_complete(Status::Invalid("Chunk cursor is shut down"));
        continue;
      }

      const auto service_start = Clock::now();
      TableResult result = Status::Invalid("chunk cursor trail miss");
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
        if (task->track_request) {
          CompleteRequest(task->request_start, true);
        }
        task->on_complete(std::move(result));
        continue;
      }

      result = [&]() -> TableResult {
        try {
          auto chunk_result = task->loader();
          if (!chunk_result.has_error()) {
            std::lock_guard<std::mutex> lock(mutex_);
            InsertTrailLocked(task->key, chunk_result.value());
          }
          return chunk_result;
        } catch (const std::exception& e) {
          return Status::UnknownError("Chunk cursor read threw: ", e.what());
        } catch (...) {
          return Status::UnknownError("Chunk cursor read threw unknown error");
        }
      }();

      chunks_read_.fetch_add(1, std::memory_order_relaxed);
      if (!result.has_error()) {
        chunks_served_.fetch_add(1, std::memory_order_relaxed);
      }

      const uint64_t service_ms = MillisecondsSince(service_start);
      service_ms_sum_.fetch_add(service_ms, std::memory_order_relaxed);
      AtomicMax(&service_ms_max_, service_ms);
      if (task->track_request) {
        CompleteRequest(task->request_start, !result.has_error());
      }
      task->on_complete(std::move(result));
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
  const bool auto_track_requests_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool shutdown_ = false;
  std::vector<std::thread> workers_;
  std::vector<QueueState> queues_;
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
  std::atomic<uint64_t> chunk_order_wraps_{0};
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

OrderedChunkCursor::OrderedChunkCursor(size_t cursor_count,
                                       size_t trail_capacity_chunks,
                                       bool auto_track_requests)
    : impl_(std::make_unique<Impl>(cursor_count, trail_capacity_chunks,
                                   auto_track_requests)) {}

OrderedChunkCursor::~OrderedChunkCursor() = default;

OrderedChunkCursor::TableResult OrderedChunkCursor::LoadChunk(
    const ChunkReadKey& key, OrderKey order_key, const TableLoader& loader) {
  return impl_->LoadChunk(key, order_key, loader);
}

Status OrderedChunkCursor::SubmitChunk(const ChunkReadKey& key,
                                       OrderKey order_key,
                                       const TableLoader& loader,
                                       CompletionCallback on_complete) {
  return impl_->SubmitChunk(key, order_key, loader, std::move(on_complete));
}

FeatureCursorStats OrderedChunkCursor::stats() const { return impl_->stats(); }

void OrderedChunkCursor::Shutdown() { impl_->Shutdown(); }

void OrderedChunkCursor::RegisterRequest() { impl_->RegisterRequest(); }

void OrderedChunkCursor::CompleteRequest(Clock::time_point start, bool ok) {
  impl_->CompleteRequest(start, ok);
}

void OrderedChunkCursor::RecordBatchServed(size_t rows) {
  impl_->RecordBatchServed(rows);
}

}  // namespace graphar::ml
