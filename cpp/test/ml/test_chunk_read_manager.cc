#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "arrow/api.h"
#include "graphar/graph_info.h"
#include "graphar/ml/chunk_read_manager.h"
#include "graphar/types.h"

#include <catch2/catch_test_macros.hpp>

#include "../util.h"

namespace graphar::ml {
namespace {

constexpr const char* kGraphPath = "/ldbc_sample/parquet/ldbc_sample.graph.yml";
constexpr const char* kVertexType = "person";
constexpr const char* kEdgeType = "knows";

ChunkReadKey TestKey(IdType chunk_id = 0) {
  ChunkReadKey key;
  key.graph_prefix = "graph";
  key.vertex_type = "node";
  key.property_group_prefix = "features";
  key.chunk_id = chunk_id;
  return key;
}

std::shared_ptr<arrow::Table> MakeTable(int64_t value) {
  arrow::Int64Builder builder;
  REQUIRE(builder.Append(value).ok());
  std::shared_ptr<arrow::Array> array;
  REQUIRE(builder.Finish(&array).ok());
  return arrow::Table::Make(
      arrow::schema({arrow::field("value", arrow::int64())}), {array});
}

size_t TableBytes(const std::shared_ptr<arrow::Table>& table) {
  size_t bytes = 0;
  for (int col = 0; col < table->num_columns(); ++col) {
    for (const auto& chunk : table->column(col)->chunks()) {
      for (const auto& buffer : chunk->data()->buffers) {
        if (buffer != nullptr) {
          bytes += static_cast<size_t>(buffer->size());
        }
      }
    }
  }
  return bytes;
}

std::shared_ptr<GraphInfo> LoadLdbcSampleGraph(const std::string& test_data_dir) {
  auto maybe_graph_info = GraphInfo::Load(test_data_dir + kGraphPath);
  REQUIRE(maybe_graph_info.status().ok());
  return maybe_graph_info.value();
}

}  // namespace

TEST_CASE("ChunkReadManager singleflight shares one load") {
  ChunkReadManager manager;
  auto table = MakeTable(42);
  std::atomic<int> loader_calls{0};

  constexpr int kThreads = 8;
  std::promise<void> start_promise;
  auto start = start_promise.get_future().share();
  std::vector<ChunkReadManager::TableResult> results(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i]() {
      start.wait();
      results[i] = manager.GetOrLoad(TestKey(), [&]() {
        loader_calls.fetch_add(1, std::memory_order_relaxed);
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (manager.stats().waiters < kThreads - 1 &&
               std::chrono::steady_clock::now() < deadline) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return table;
      });
    });
  }

  start_promise.set_value();
  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(loader_calls.load() == 1);
  for (const auto& result : results) {
    REQUIRE(result.status().ok());
    REQUIRE(result.value() == table);
  }

  const auto stats = manager.stats();
  REQUIRE(stats.requests == kThreads);
  REQUIRE(stats.leaders == 1);
  REQUIRE(stats.waiters == kThreads - 1);
  REQUIRE(stats.completed == kThreads);
  REQUIRE(stats.failed == 0);
}

TEST_CASE("ChunkReadManager does not merge distinct keys") {
  ChunkReadManager manager;
  std::atomic<int> loader_calls{0};

  auto first = manager.GetOrLoad(TestKey(1), [&]() {
    loader_calls.fetch_add(1, std::memory_order_relaxed);
    return MakeTable(1);
  });
  auto second = manager.GetOrLoad(TestKey(2), [&]() {
    loader_calls.fetch_add(1, std::memory_order_relaxed);
    return MakeTable(2);
  });

  REQUIRE(first.status().ok());
  REQUIRE(second.status().ok());
  REQUIRE(loader_calls.load() == 2);

  const auto stats = manager.stats();
  REQUIRE(stats.requests == 2);
  REQUIRE(stats.leaders == 2);
  REQUIRE(stats.waiters == 0);
  REQUIRE(stats.completed == 2);
  REQUIRE(stats.failed == 0);
}

TEST_CASE("ChunkReadManager serves later requests from RAM cache") {
  ChunkReadManagerOptions options;
  options.ram_budget_bytes = 1024;
  ChunkReadManager manager(options);
  auto table = MakeTable(11);
  std::atomic<int> loader_calls{0};

  auto first = manager.GetOrLoad(TestKey(), [&]() {
    loader_calls.fetch_add(1, std::memory_order_relaxed);
    return table;
  });
  auto second = manager.GetOrLoad(TestKey(), [&]() {
    loader_calls.fetch_add(1, std::memory_order_relaxed);
    return MakeTable(12);
  });

  REQUIRE(first.status().ok());
  REQUIRE(second.status().ok());
  REQUIRE(first.value() == table);
  REQUIRE(second.value() == table);
  REQUIRE(loader_calls.load() == 1);

  const auto stats = manager.stats();
  REQUIRE(stats.requests == 2);
  REQUIRE(stats.leaders == 1);
  REQUIRE(stats.ram_cache_hits == 1);
  REQUIRE(stats.ram_cache_misses == 1);
  REQUIRE(stats.ram_cache_evictions == 0);
  REQUIRE(stats.ram_cache_bytes >= TableBytes(table));
}

TEST_CASE("ChunkReadManager zero RAM budget disables cache") {
  ChunkReadManager manager;
  std::atomic<int> loader_calls{0};

  auto first = manager.GetOrLoad(TestKey(), [&]() {
    loader_calls.fetch_add(1, std::memory_order_relaxed);
    return MakeTable(1);
  });
  auto second = manager.GetOrLoad(TestKey(), [&]() {
    loader_calls.fetch_add(1, std::memory_order_relaxed);
    return MakeTable(2);
  });

  REQUIRE(first.status().ok());
  REQUIRE(second.status().ok());
  REQUIRE(loader_calls.load() == 2);

  const auto stats = manager.stats();
  REQUIRE(stats.ram_cache_hits == 0);
  REQUIRE(stats.ram_cache_misses == 0);
  REQUIRE(stats.ram_cache_bytes == 0);
}

TEST_CASE("ChunkReadManager evicts least recently used chunks") {
  auto table = MakeTable(1);
  ChunkReadManagerOptions options;
  options.ram_budget_bytes = TableBytes(table) * 2;
  ChunkReadManager manager(options);
  std::atomic<int> loader_calls{0};

  REQUIRE(manager.GetOrLoad(TestKey(1), [&]() {
                   loader_calls.fetch_add(1, std::memory_order_relaxed);
                   return MakeTable(1);
                 })
              .status()
              .ok());
  REQUIRE(manager.GetOrLoad(TestKey(2), [&]() {
                   loader_calls.fetch_add(1, std::memory_order_relaxed);
                   return MakeTable(2);
                 })
              .status()
              .ok());
  REQUIRE(manager.GetOrLoad(TestKey(1), [&]() {
                   loader_calls.fetch_add(1, std::memory_order_relaxed);
                   return MakeTable(10);
                 })
              .status()
              .ok());
  REQUIRE(manager.GetOrLoad(TestKey(3), [&]() {
                   loader_calls.fetch_add(1, std::memory_order_relaxed);
                   return MakeTable(3);
                 })
              .status()
              .ok());
  REQUIRE(manager.GetOrLoad(TestKey(2), [&]() {
                   loader_calls.fetch_add(1, std::memory_order_relaxed);
                   return MakeTable(20);
                 })
              .status()
              .ok());

  REQUIRE(loader_calls.load() == 4);
  const auto stats = manager.stats();
  REQUIRE(stats.ram_cache_hits == 1);
  REQUIRE(stats.ram_cache_evictions >= 1);
  REQUIRE(stats.ram_cache_bytes <= options.ram_budget_bytes);
}

TEST_CASE("ChunkReadManager does not cache chunks larger than RAM budget") {
  auto table = MakeTable(1);
  ChunkReadManagerOptions options;
  options.ram_budget_bytes = TableBytes(table) - 1;
  ChunkReadManager manager(options);
  std::atomic<int> loader_calls{0};

  auto first = manager.GetOrLoad(TestKey(), [&]() {
    loader_calls.fetch_add(1, std::memory_order_relaxed);
    return table;
  });
  auto second = manager.GetOrLoad(TestKey(), [&]() {
    loader_calls.fetch_add(1, std::memory_order_relaxed);
    return table;
  });

  REQUIRE(first.status().ok());
  REQUIRE(second.status().ok());
  REQUIRE(loader_calls.load() == 2);

  const auto stats = manager.stats();
  REQUIRE(stats.ram_cache_hits == 0);
  REQUIRE(stats.ram_cache_misses == 2);
  REQUIRE(stats.ram_cache_bytes == 0);
}

TEST_CASE_METHOD(GlobalFixture, "ChunkReadManager caches edge offset chunks") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  ChunkReadManagerOptions options;
  options.ram_budget_bytes = 1024 * 1024;
  ChunkReadManager manager(options);

  auto first = manager.GetEdgeOffsetChunk(
      graph_info, kVertexType, kEdgeType, kVertexType,
      AdjListType::ordered_by_source, /*vertex_chunk_id=*/0);
  auto second = manager.GetEdgeOffsetChunk(
      graph_info, kVertexType, kEdgeType, kVertexType,
      AdjListType::ordered_by_source, /*vertex_chunk_id=*/0);

  REQUIRE(first.status().ok());
  REQUIRE(second.status().ok());
  REQUIRE(first.value()->Equals(*second.value()));

  const auto stats = manager.stats();
  REQUIRE(stats.requests == 2);
  REQUIRE(stats.leaders == 1);
  REQUIRE(stats.ram_cache_hits == 1);
  REQUIRE(stats.ram_cache_misses == 1);
  REQUIRE(stats.failed == 0);
}

TEST_CASE_METHOD(GlobalFixture, "ChunkReadManager caches edge adjacency chunks") {
  auto graph_info = LoadLdbcSampleGraph(test_data_dir);
  ChunkReadManagerOptions options;
  options.ram_budget_bytes = 1024 * 1024;
  ChunkReadManager manager(options);

  auto first = manager.GetEdgeAdjListChunk(
      graph_info, kVertexType, kEdgeType, kVertexType,
      AdjListType::ordered_by_source, /*vertex_chunk_id=*/0, /*chunk_id=*/0);
  auto second = manager.GetEdgeAdjListChunk(
      graph_info, kVertexType, kEdgeType, kVertexType,
      AdjListType::ordered_by_source, /*vertex_chunk_id=*/0, /*chunk_id=*/0);

  REQUIRE(first.status().ok());
  REQUIRE(second.status().ok());
  REQUIRE(first.value() != nullptr);
  REQUIRE(first.value()->Equals(*second.value()));

  const auto stats = manager.stats();
  REQUIRE(stats.requests == 2);
  REQUIRE(stats.leaders == 1);
  REQUIRE(stats.ram_cache_hits == 1);
  REQUIRE(stats.ram_cache_misses == 1);
  REQUIRE(stats.failed == 0);
}

TEST_CASE("ChunkReadManager propagates failures and allows retry") {
  ChunkReadManager manager;
  std::atomic<int> loader_calls{0};

  auto failed = manager.GetOrLoad(TestKey(), [&]() {
    loader_calls.fetch_add(1, std::memory_order_relaxed);
    return Status::Invalid("synthetic failure");
  });
  REQUIRE(failed.has_error());

  auto retried = manager.GetOrLoad(TestKey(), [&]() {
    loader_calls.fetch_add(1, std::memory_order_relaxed);
    return MakeTable(7);
  });
  REQUIRE(retried.status().ok());

  REQUIRE(loader_calls.load() == 2);
  const auto stats = manager.stats();
  REQUIRE(stats.requests == 2);
  REQUIRE(stats.leaders == 2);
  REQUIRE(stats.completed == 1);
  REQUIRE(stats.failed == 1);
}

}  // namespace graphar::ml
