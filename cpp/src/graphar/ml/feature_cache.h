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
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "graphar/types.h"

namespace arrow {
class Scalar;
class Table;
}

namespace graphar::ml {

/**
 * Thread-safe LFU cache for individual node feature rows fetched by
 * GetNodeFeatures.
 *
 * Cache key: (graph_info*, property_group*, node_id). Each entry stores one
 * property-group row for that node.
 * Multiple threads sharing the same FeatureCache instance benefit from
 * shared hits — all synchronization is internal.
 *
 * Lifetime is independent of any loader. A single cache can be shared across
 * multiple loaders (e.g. train/val/test) and lives as long as the caller
 * holds a reference.
 */
class FeatureCache {
 public:
  struct CachedRow {
    std::vector<std::shared_ptr<arrow::Scalar>> values;
    size_t size_bytes{0};
  };

  explicit FeatureCache(size_t max_bytes);

  /**
   * Look up a cached row. Returns nullptr on miss.
   * On hit, promotes the entry's frequency (O(1) LFU).
   */
  std::shared_ptr<const CachedRow> Get(const void* graph_info_ptr,
                                       const void* pg_ptr, IdType node_id);

  /**
   * Insert a cached row. Evicts LFU entries if needed to stay within budget.
   * No-op if the node already exists or its row is larger than the total budget.
   */
  void Put(const void* graph_info_ptr, const void* pg_ptr, IdType node_id,
           std::shared_ptr<CachedRow> row);

  /** Remove all cached entries. */
  void Clear();

  size_t size_bytes() const;
  size_t num_nodes() const;
  size_t max_bytes() const { return max_bytes_; }
  size_t hits() const { return hits_.load(); }
  size_t misses() const { return misses_.load(); }
  double hit_rate() const;

  size_t chunks_read() const { return chunks_read_.load(); }
  size_t chunks_skipped() const { return chunks_skipped_.load(); }
  double io_saved_pct() const;

  /** Called by GetNodeFeatures to record chunk-level I/O outcomes. */
  void RecordChunksRead(size_t n);
  void RecordChunksSkipped(size_t n);

 private:
  struct CacheKey {
    const void* graph_info_ptr;
    const void* pg_ptr;
    IdType node_id;

    bool operator==(const CacheKey& o) const noexcept {
      return graph_info_ptr == o.graph_info_ptr && pg_ptr == o.pg_ptr &&
             node_id == o.node_id;
    }
  };

  struct CacheKeyHash {
    size_t operator()(const CacheKey& k) const noexcept {
      size_t h = std::hash<const void*>{}(k.graph_info_ptr);
      h ^= std::hash<const void*>{}(k.pg_ptr) + 0x9e3779b9u + (h << 6) +
           (h >> 2);
      h ^= std::hash<IdType>{}(k.node_id) + 0x9e3779b9u + (h << 6) +
           (h >> 2);
      return h;
    }
  };

  struct Entry {
    std::shared_ptr<CachedRow> row;
    size_t freq;
    size_t size_bytes;
    std::list<CacheKey>::iterator list_it;  // position in freq_to_keys_[freq]
  };

  // Evict the single LFU entry. Caller must hold mutex_.
  void EvictOne();

  std::unordered_map<CacheKey, Entry, CacheKeyHash> key_to_entry_;
  std::unordered_map<size_t, std::list<CacheKey>> freq_to_keys_;
  size_t min_freq_{0};
  size_t current_bytes_{0};
  const size_t max_bytes_;
  bool full_logged_{false};

  mutable std::mutex mutex_;
  std::atomic<size_t> hits_{0};
  std::atomic<size_t> misses_{0};
  std::atomic<size_t> chunks_read_{0};
  std::atomic<size_t> chunks_skipped_{0};
};

}  // namespace graphar::ml
