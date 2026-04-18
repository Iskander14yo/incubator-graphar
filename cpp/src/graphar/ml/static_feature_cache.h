#pragma once

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "graphar/fwd.h"

namespace arrow {
class Table;
class ChunkedArray;
}

namespace graphar {

class GraphInfo;

namespace ml {

class StaticFeatureCache {
 public:
  explicit StaticFeatureCache(std::shared_ptr<GraphInfo> graph_info);

  Status Pin(const std::string& vertex_type,
             const std::vector<IdType>& node_ids,
             const std::vector<std::string>& properties);

  struct LookupResult {
    std::shared_ptr<arrow::Table> hits;
    std::vector<size_t> hit_positions;
    std::vector<size_t> miss_positions;
    std::vector<IdType> miss_node_ids;
  };

  Result<LookupResult> Lookup(const std::string& vertex_type,
                              const std::vector<IdType>& node_ids,
                              const std::vector<std::string>& properties);

  size_t num_nodes() const;
  size_t size_bytes() const;
  size_t hits() const;
  size_t misses() const;
  double hit_rate() const;

 private:
  size_t ComputeStorageBytes() const;
  size_t ComputeStorageBytesUnlocked() const;

  std::shared_ptr<GraphInfo> graph_info_;
  mutable std::shared_mutex mu_;

  struct PgPin {
    std::vector<std::string> props_order;
    std::vector<IdType> rowid_order;
    std::unordered_map<IdType, int64_t> id_to_row;
    std::unordered_map<std::string, std::shared_ptr<arrow::ChunkedArray>> columns;
  };

  struct PgHash {
    size_t operator()(const std::shared_ptr<PropertyGroup>& p) const noexcept {
      return std::hash<PropertyGroup*>{}(p.get());
    }
  };
  struct PgEq {
    bool operator()(const std::shared_ptr<PropertyGroup>& a,
                    const std::shared_ptr<PropertyGroup>& b) const noexcept {
      return a.get() == b.get();
    }
  };

  std::unordered_map<
      std::string,
      std::unordered_map<std::shared_ptr<PropertyGroup>, PgPin, PgHash, PgEq>>
      pins_;

  std::unordered_set<IdType> pinned_ids_;
  std::atomic<size_t> hits_{0};
  std::atomic<size_t> misses_{0};
};

}  // namespace ml
}  // namespace graphar
