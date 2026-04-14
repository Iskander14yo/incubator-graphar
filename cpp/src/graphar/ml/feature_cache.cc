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

#include "graphar/ml/feature_cache.h"

#include <algorithm>
#include <cstdio>

namespace graphar::ml {

FeatureCache::FeatureCache(size_t max_bytes) : max_bytes_(max_bytes) {}

std::shared_ptr<const FeatureCache::CachedRow> FeatureCache::Get(
    const void* graph_info_ptr, const void* pg_ptr, IdType node_id) {
  CacheKey key{graph_info_ptr, pg_ptr, node_id};
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = key_to_entry_.find(key);
  if (it == key_to_entry_.end()) {
    misses_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }

  hits_.fetch_add(1, std::memory_order_relaxed);
  Entry& entry = it->second;
  size_t old_freq = entry.freq;

  // Remove from current frequency list
  freq_to_keys_[old_freq].erase(entry.list_it);
  if (freq_to_keys_[old_freq].empty()) {
    freq_to_keys_.erase(old_freq);
    if (min_freq_ == old_freq) ++min_freq_;
  }

  // Promote to next frequency
  entry.freq = old_freq + 1;
  auto& new_list = freq_to_keys_[entry.freq];
  new_list.push_front(key);
  entry.list_it = new_list.begin();

  return entry.row;
}

void FeatureCache::Put(const void* graph_info_ptr, const void* pg_ptr,
                       IdType node_id, std::shared_ptr<CachedRow> row) {
  if (row == nullptr) return;
  size_t size = row->size_bytes;
  if (size > max_bytes_) return;  // will never fit; skip silently

  CacheKey key{graph_info_ptr, pg_ptr, node_id};
  std::lock_guard<std::mutex> lock(mutex_);

  if (key_to_entry_.count(key)) return;  // already cached

  if (current_bytes_ + size > max_bytes_) {
    if (!full_logged_) {
      full_logged_ = true;
      std::fprintf(stderr,
                   "[graphar::ml::FeatureCache] Cache full "
                   "(%.1f / %.1f MB, %zu nodes cached). "
                   "LFU eviction starting.\n",
                   static_cast<double>(current_bytes_) / (1024.0 * 1024.0),
                   static_cast<double>(max_bytes_) / (1024.0 * 1024.0),
                   key_to_entry_.size());
    }
    while (current_bytes_ + size > max_bytes_ && !key_to_entry_.empty()) {
      EvictOne();
    }
  }

  // Insert with frequency 1 at the front (most-recently-used position)
  auto& list = freq_to_keys_[1];
  list.push_front(key);
  key_to_entry_[key] = Entry{std::move(row), 1, size, list.begin()};
  current_bytes_ += size;
  min_freq_ = 1;
}

void FeatureCache::EvictOne() {
  // Evict from the least-frequently-used bucket (LRU end of that bucket)
  auto& min_list = freq_to_keys_[min_freq_];
  CacheKey evict_key = min_list.back();
  min_list.pop_back();

  if (min_list.empty()) {
    freq_to_keys_.erase(min_freq_);
    // Rescan for new min_freq_ to avoid UB on next eviction call
    if (!freq_to_keys_.empty()) {
      min_freq_ = std::min_element(
                      freq_to_keys_.begin(), freq_to_keys_.end(),
                      [](const auto& a, const auto& b) {
                        return a.first < b.first;
                      })
                      ->first;
    }
  }

  auto it = key_to_entry_.find(evict_key);
  current_bytes_ -= it->second.size_bytes;
  key_to_entry_.erase(it);
}

void FeatureCache::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  key_to_entry_.clear();
  freq_to_keys_.clear();
  min_freq_ = 0;
  current_bytes_ = 0;
  full_logged_ = false;
}

size_t FeatureCache::size_bytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return current_bytes_;
}

size_t FeatureCache::num_nodes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return key_to_entry_.size();
}

double FeatureCache::hit_rate() const {
  size_t h = hits_.load(std::memory_order_relaxed);
  size_t m = misses_.load(std::memory_order_relaxed);
  size_t total = h + m;
  return total == 0 ? 0.0 : static_cast<double>(h) / static_cast<double>(total);
}

double FeatureCache::io_saved_pct() const {
  size_t r = chunks_read_.load(std::memory_order_relaxed);
  size_t s = chunks_skipped_.load(std::memory_order_relaxed);
  size_t total = r + s;
  return total == 0 ? 0.0 : 100.0 * static_cast<double>(s) / static_cast<double>(total);
}

void FeatureCache::RecordChunksRead(size_t n) {
  chunks_read_.fetch_add(n, std::memory_order_relaxed);
}

void FeatureCache::RecordChunksSkipped(size_t n) {
  chunks_skipped_.fetch_add(n, std::memory_order_relaxed);
}

}  // namespace graphar::ml
