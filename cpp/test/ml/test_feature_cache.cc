#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "graphar/graph_info.h"
#include "graphar/ml/feature_cache.h"
#include "graphar/ml/neighbor_sampling.h"

#include <catch2/catch_test_macros.hpp>

#include "../util.h"

namespace graphar::ml {
namespace {

// ---- helpers ----------------------------------------------------------------

std::shared_ptr<arrow::Table> MakeInt64Table(int64_t num_rows) {
  arrow::Int64Builder builder;
  (void)builder.Reserve(num_rows);
  for (int64_t i = 0; i < num_rows; ++i) builder.UnsafeAppend(i);
  std::shared_ptr<arrow::Array> arr;
  (void)builder.Finish(&arr);
  return arrow::Table::Make(
      arrow::schema({arrow::field("v", arrow::int64())}),
      {std::make_shared<arrow::ChunkedArray>(arr)}, num_rows);
}

size_t ArrowTableBytes(const arrow::Table& t) {
  size_t total = 0;
  for (int i = 0; i < t.num_columns(); ++i)
    for (const auto& chunk : t.column(i)->chunks())
      for (const auto& buf : chunk->data()->buffers)
        if (buf) total += static_cast<size_t>(buf->size());
  return total;
}

std::vector<int64_t> Int64Values(const std::shared_ptr<arrow::ChunkedArray>& col) {
  std::vector<int64_t> out;
  out.reserve(col->length());
  for (const auto& chunk : col->chunks()) {
    auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
    for (int64_t i = 0; i < arr->length(); ++i) out.push_back(arr->Value(i));
  }
  return out;
}

// Stable fake pointer constants for cache keys
void* const kG  = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000));
void* const kPg = reinterpret_cast<void*>(static_cast<uintptr_t>(0x2000));

}  // namespace

// =============================================================================
// LFU unit tests (no graph I/O needed)
// =============================================================================

TEST_CASE("FeatureCache - miss on empty cache") {
  FeatureCache cache(1024 * 1024);
  REQUIRE(cache.Get(kG, kPg, 0) == nullptr);
  REQUIRE(cache.misses() == 1);
  REQUIRE(cache.hits() == 0);
  REQUIRE(cache.num_nodes() == 0);
  REQUIRE(cache.size_bytes() == 0);
  REQUIRE(cache.hit_rate() == 0.0);
}

TEST_CASE("FeatureCache - put then get returns same table") {
  auto table = MakeInt64Table(100);
  FeatureCache cache(ArrowTableBytes(*table) * 4);

  cache.Put(kG, kPg, 0, table);
  REQUIRE(cache.num_nodes() == 1);
  REQUIRE(cache.size_bytes() > 0);

  auto got = cache.Get(kG, kPg, 0);
  REQUIRE(got != nullptr);
  REQUIRE(got.get() == table.get());  // same underlying object
  REQUIRE(cache.hits() == 1);
  REQUIRE(cache.misses() == 0);
}

TEST_CASE("FeatureCache - duplicate put is no-op") {
  auto t1 = MakeInt64Table(100);
  auto t2 = MakeInt64Table(100);
  FeatureCache cache(ArrowTableBytes(*t1) * 4);

  cache.Put(kG, kPg, 0, t1);
  cache.Put(kG, kPg, 0, t2);  // same key — should be ignored

  REQUIRE(cache.num_nodes() == 1);
  REQUIRE(cache.Get(kG, kPg, 0).get() == t1.get());  // t1 wins
}

TEST_CASE("FeatureCache - LFU evicts least-frequent entry") {
  // Put A and B (both freq=1), promote A twice, then insert C.
  // B has lower frequency so B is evicted, A and C survive.
  auto ta = MakeInt64Table(100);
  auto tb = MakeInt64Table(100);
  auto tc = MakeInt64Table(100);
  size_t sz = ArrowTableBytes(*ta);
  FeatureCache cache(2 * sz);  // fits exactly 2 entries

  cache.Put(kG, kPg, 0, ta);  // A: freq=1
  cache.Put(kG, kPg, 1, tb);  // B: freq=1
  cache.Get(kG, kPg, 0);       // A: freq=2
  cache.Get(kG, kPg, 0);       // A: freq=3
  // min_freq=1 → only B qualifies; Put(C) evicts B

  cache.Put(kG, kPg, 2, tc);

  REQUIRE(cache.Get(kG, kPg, 1) == nullptr);  // B evicted
  REQUIRE(cache.Get(kG, kPg, 0) != nullptr);  // A kept
  REQUIRE(cache.Get(kG, kPg, 2) != nullptr);  // C present
  REQUIRE(cache.num_nodes() == 2);
}

TEST_CASE("FeatureCache - LRU tie-break within same frequency") {
  // Put A first, then B — both freq=1. A is the LRU (oldest at back of list).
  // Inserting C must evict A, leaving B and C.
  auto ta = MakeInt64Table(100);
  auto tb = MakeInt64Table(100);
  auto tc = MakeInt64Table(100);
  size_t sz = ArrowTableBytes(*ta);
  FeatureCache cache(2 * sz);

  cache.Put(kG, kPg, 0, ta);  // A: inserted first → LRU end of freq=1
  cache.Put(kG, kPg, 1, tb);  // B: inserted second → MRU end of freq=1

  cache.Put(kG, kPg, 2, tc);  // evicts A (LRU)

  REQUIRE(cache.Get(kG, kPg, 0) == nullptr);  // A evicted
  REQUIRE(cache.Get(kG, kPg, 1) != nullptr);  // B kept
  REQUIRE(cache.Get(kG, kPg, 2) != nullptr);  // C present
}

TEST_CASE("FeatureCache - clear empties entries, preserves stats") {
  auto t = MakeInt64Table(100);
  FeatureCache cache(ArrowTableBytes(*t) * 4);

  cache.Put(kG, kPg, 0, t);
  cache.Get(kG, kPg, 0);  // hit
  cache.Get(kG, kPg, 1);  // miss

  cache.Clear();

  REQUIRE(cache.num_nodes() == 0);
  REQUIRE(cache.size_bytes() == 0);
  REQUIRE(cache.Get(kG, kPg, 0) == nullptr);  // evicted by Clear
  // cumulative stats preserved
  REQUIRE(cache.hits() == 1);
  REQUIRE(cache.misses() == 2);  // 1 before + 1 after clear
}

TEST_CASE("FeatureCache - hit_rate") {
  auto t = MakeInt64Table(100);
  FeatureCache cache(ArrowTableBytes(*t) * 4);

  REQUIRE(cache.hit_rate() == 0.0);  // no lookups yet

  cache.Put(kG, kPg, 0, t);
  cache.Get(kG, kPg, 0);  // hit
  cache.Get(kG, kPg, 0);  // hit
  cache.Get(kG, kPg, 1);  // miss

  REQUIRE(cache.hits() == 2);
  REQUIRE(cache.misses() == 1);
  REQUIRE(std::abs(cache.hit_rate() - 2.0 / 3.0) < 1e-9);
}

TEST_CASE("FeatureCache - entry too large for budget is silently dropped") {
  auto t = MakeInt64Table(1000);
  size_t sz = ArrowTableBytes(*t);
  FeatureCache cache(sz / 2);  // budget too small for one entry

  cache.Put(kG, kPg, 0, t);  // must be silently ignored

  REQUIRE(cache.num_nodes() == 0);
  REQUIRE(cache.Get(kG, kPg, 0) == nullptr);
}

// =============================================================================
// Integration tests: FeatureCache + GetNodeFeatures
// =============================================================================

TEST_CASE_METHOD(GlobalFixture, "GetNodeFeatures - node-cached result equals uncached") {
  auto maybe_graph = GraphInfo::Load(test_data_dir + "/ldbc_sample/parquet/ldbc_sample.graph.yml");
  REQUIRE(maybe_graph.status().ok());
  auto graph = maybe_graph.value();

  FeatureCache cache(64 * 1024 * 1024);  // 64 MB — no eviction expected
  std::vector<IdType> node_ids = {0, 1, 2, 5, 10};

  auto r_plain  = GetNodeFeatures(graph, "person", node_ids, {"id"});
  auto r_cached = GetNodeFeatures(graph, "person", node_ids, {"id"}, &cache);
  REQUIRE(r_plain.status().ok());
  REQUIRE(r_cached.status().ok());

  REQUIRE(Int64Values(r_plain.value()->GetColumnByName("id")) ==
          Int64Values(r_cached.value()->GetColumnByName("id")));
}

TEST_CASE_METHOD(GlobalFixture, "GetNodeFeatures - second call hits per-node cache") {
  auto maybe_graph = GraphInfo::Load(test_data_dir + "/ldbc_sample/parquet/ldbc_sample.graph.yml");
  REQUIRE(maybe_graph.status().ok());
  auto graph = maybe_graph.value();

  FeatureCache cache(64 * 1024 * 1024);
  std::vector<IdType> node_ids = {0, 1, 2};

  // First call — all misses
  auto r1 = GetNodeFeatures(graph, "person", node_ids, {"id"}, &cache);
  REQUIRE(r1.status().ok());
  REQUIRE(cache.misses() > 0);
  REQUIRE(cache.hits() == 0);
  size_t nodes_after_first = cache.num_nodes();
  size_t misses_after_first = cache.misses();

  // Second call — same nodes, same props → all chunks already in cache
  auto r2 = GetNodeFeatures(graph, "person", node_ids, {"id"}, &cache);
  REQUIRE(r2.status().ok());
  REQUIRE(cache.hits() > 0);
  REQUIRE(cache.misses() == misses_after_first);  // no new misses
  REQUIRE(cache.num_nodes() == nodes_after_first);  // same nodes, no new inserts

  REQUIRE(Int64Values(r1.value()->GetColumnByName("id")) ==
          Int64Values(r2.value()->GetColumnByName("id")));
}

}  // namespace graphar::ml
