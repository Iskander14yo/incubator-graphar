#include "graphar/ml/neighbor_sampling.h"

#include <algorithm>

#include "graphar/ml/feature_cache.h"
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "arrow/api.h"
#include "arrow/compute/api.h"
#include "graphar/arrow/chunk_reader.h"
#include "graphar/filesystem.h"
#include "graphar/graph_info.h"
#include "graphar/result.h"
#include "graphar/types.h"

namespace graphar::ml {

namespace {

void AppendReservoirSample(std::vector<IdType>* neighbors, IdType* seen_neighbors,
                           const int64_t* values, IdType count, int max_neighbors,
                           std::mt19937* gen) {
  if (max_neighbors == 0) {
    *seen_neighbors += count;
    return;
  }

  for (IdType i = 0; i < count; ++i) {
    IdType current = *seen_neighbors;
    if (neighbors->size() < static_cast<size_t>(max_neighbors)) {
      neighbors->push_back(values[i]);
    } else {
      std::uniform_int_distribution<IdType> dist(0, current);
      IdType replace_idx = dist(*gen);
      if (replace_idx < max_neighbors) {
        (*neighbors)[replace_idx] = values[i];
      }
    }
    *seen_neighbors = current + 1;
  }
}

class Int64ChunkedArrayCursor {
 public:
  explicit Int64ChunkedArrayCursor(
      std::shared_ptr<arrow::ChunkedArray> chunked_array)
      : chunked_array_(std::move(chunked_array)),
        chunk_index_(0),
        chunk_begin_(0),
        chunk_length_(0) {
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
    IdType local_row = row - chunk_begin_;
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
          chunked_array_->chunk(chunk_index_));
      chunk_length_ = current_chunk_->length();
    }
  }

  std::shared_ptr<arrow::ChunkedArray> chunked_array_;
  int chunk_index_;
  IdType chunk_begin_;
  IdType chunk_length_;
  std::shared_ptr<arrow::Int64Array> current_chunk_;
};

Result<std::shared_ptr<arrow::ChunkedArray>> LoadOffsetColumn(
    const std::shared_ptr<FileSystem>& fs, const std::string& prefix,
    const std::shared_ptr<EdgeInfo>& edge_info, AdjListType adj_list_type,
    IdType vertex_chunk_idx) {
  GAR_ASSIGN_OR_RAISE(
      auto chunk_file_path,
      edge_info->GetAdjListOffsetFilePath(vertex_chunk_idx, adj_list_type));
  auto file_type = edge_info->GetAdjacentList(adj_list_type)->GetFileType();
  GAR_ASSIGN_OR_RAISE(auto table,
                      fs->ReadFileToTable(prefix + chunk_file_path, file_type));
  if (table->num_columns() == 0) {
    return Status::Invalid("Offset file for edge type '",
                           edge_info->GetEdgeType(), "' has no columns");
  }
  return table->column(0);
}

}  // namespace

Result<SamplingResult> SampleNeighbors(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::string& vertex_type, const std::string& edge_type,
    const std::vector<IdType>& seed_nodes, const std::vector<int>& fanout,
    uint64_t seed) {
  if (seed_nodes.empty()) {
    SamplingResult result;
    result.num_sampled_nodes_per_hop.assign(fanout.size() + 1, 0);
    result.num_sampled_edges_per_hop.assign(fanout.size(), 0);
    return result;
  }
  if (fanout.empty()) return Status::Invalid("Fanout cannot be empty");
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

  const std::string& prefix = graph_info->GetPrefix();
  std::string normalized_prefix;
  GAR_ASSIGN_OR_RAISE(auto fs,
                      FileSystemFromUriOrPath(prefix, &normalized_prefix));
  const auto adj_type = AdjListType::ordered_by_source;
  const IdType src_chunk_size = edge_info->GetSrcChunkSize();

  // Persistent reader — chunk caches survive across seeks within the same
  // chunk, unlike EdgesCollection/EdgeIter which re-create readers per source
  // node.
  AdjListArrowChunkReader adj_reader(edge_info, adj_type, prefix);

  std::unordered_set<IdType> all_sampled_nodes_set;
  std::vector<IdType> sampled_nodes;
  sampled_nodes.reserve(seed_nodes.size());
  for (IdType seed_node : seed_nodes) {
    if (all_sampled_nodes_set.insert(seed_node).second) {
      sampled_nodes.push_back(seed_node);
    }
  }
  std::vector<std::pair<IdType, IdType>> edge_list;
  std::vector<IdType> current_frontier = seed_nodes;
  std::mt19937 gen(seed);
  const IdType edge_chunk_size = edge_info->GetChunkSize();
  std::vector<IdType> num_sampled_nodes_per_hop;
  num_sampled_nodes_per_hop.reserve(fanout.size() + 1);
  num_sampled_nodes_per_hop.push_back(sampled_nodes.size());
  std::vector<IdType> num_sampled_edges_per_hop;
  num_sampled_edges_per_hop.reserve(fanout.size());

  for (size_t hop = 0; hop < fanout.size(); ++hop) {
    int max_neighbors = fanout[hop];
    std::vector<IdType> next_frontier;
    IdType hop_num_sampled_nodes = 0;
    IdType hop_num_sampled_edges = 0;

    std::vector<IdType> sorted_frontier = current_frontier;
    std::sort(sorted_frontier.begin(), sorted_frontier.end());
    for (size_t fi = 0; fi < sorted_frontier.size();) {
      IdType vertex_chunk_idx = sorted_frontier[fi] / src_chunk_size;

      size_t group_end = fi + 1;
      while (group_end < sorted_frontier.size() &&
             sorted_frontier[group_end] / src_chunk_size == vertex_chunk_idx) {
        ++group_end;
      }

      GAR_ASSIGN_OR_RAISE(
          auto offset_column, LoadOffsetColumn(fs, normalized_prefix, edge_info,
                                               adj_type, vertex_chunk_idx));
      Int64ChunkedArrayCursor offset_cursor(offset_column);
      IdType offset_len = offset_column->length();

      GAR_RETURN_NOT_OK(adj_reader.seek_chunk_index(vertex_chunk_idx));

      IdType cached_edge_chunk_idx = -1;
      std::shared_ptr<arrow::ChunkedArray> cached_dst_column;
      std::shared_ptr<Int64ChunkedArrayCursor> cached_cursor;

      for (size_t si = fi; si < group_end; ++si) {
        IdType src_node = sorted_frontier[si];
        IdType local_idx = src_node - vertex_chunk_idx * src_chunk_size;
        if (local_idx + 1 >= offset_len) continue;

        GAR_ASSIGN_OR_RAISE(auto begin_off, offset_cursor.ValueAt(local_idx));
        GAR_ASSIGN_OR_RAISE(auto end_off, offset_cursor.ValueAt(local_idx + 1));
        if (begin_off >= end_off) continue;

        IdType total_edges = end_off - begin_off;
        std::vector<IdType> neighbors;
        neighbors.reserve(std::min<IdType>(total_edges, max_neighbors));
        IdType seen_neighbors = 0;

        IdType cur_off = begin_off;
        while (cur_off < end_off) {
          IdType edge_chunk_idx = cur_off / edge_chunk_size;
          IdType edge_chunk_start = edge_chunk_idx * edge_chunk_size;
          IdType edge_chunk_end =
              std::min(end_off, edge_chunk_start + edge_chunk_size);

          if (edge_chunk_idx != cached_edge_chunk_idx) {
            GAR_RETURN_NOT_OK(adj_reader.seek(edge_chunk_start));
            GAR_ASSIGN_OR_RAISE(auto chunk_table, adj_reader.GetChunk());
            if (!chunk_table) break;

            cached_dst_column = chunk_table->column(1);
            cached_cursor =
                std::make_shared<Int64ChunkedArrayCursor>(cached_dst_column);
            cached_edge_chunk_idx = edge_chunk_idx;
          }

          IdType row = cur_off - edge_chunk_start;
          IdType row_end = edge_chunk_end - edge_chunk_start;
          while (row < row_end) {
            IdType available = 0;
            GAR_ASSIGN_OR_RAISE(auto raw,
                                cached_cursor->RawValuesAt(row, &available));
            IdType batch = std::min(row_end - row, available);
            AppendReservoirSample(&neighbors, &seen_neighbors, raw, batch,
                                  max_neighbors, &gen);
            row += batch;
          }

          cur_off = edge_chunk_end;
        }

        if (total_edges > max_neighbors) {
          std::shuffle(neighbors.begin(), neighbors.end(), gen);
        }

        for (IdType dst_node : neighbors) {
          edge_list.emplace_back(src_node, dst_node);
          ++hop_num_sampled_edges;
          if (all_sampled_nodes_set.insert(dst_node).second) {
            sampled_nodes.push_back(dst_node);
            next_frontier.push_back(dst_node);
            ++hop_num_sampled_nodes;
          }
        }
      }

      fi = group_end;
    }

    num_sampled_nodes_per_hop.push_back(hop_num_sampled_nodes);
    num_sampled_edges_per_hop.push_back(hop_num_sampled_edges);
    current_frontier = std::move(next_frontier);
  }

  // Convert to result format
  SamplingResult result;

  // Create ordered list of sampled nodes
  result.sampled_nodes = std::move(sampled_nodes);
  result.num_sampled_nodes_per_hop = std::move(num_sampled_nodes_per_hop);
  result.num_sampled_edges_per_hop = std::move(num_sampled_edges_per_hop);

  // Create node ID to index mapping
  std::unordered_map<IdType, IdType> node_to_index;
  for (size_t i = 0; i < result.sampled_nodes.size(); ++i) {
    node_to_index[result.sampled_nodes[i]] = i;
  }

  // Convert edges to indices
  result.src_indices.reserve(edge_list.size());
  result.dst_indices.reserve(edge_list.size());
  for (const auto& [src, dst] : edge_list) {
    result.src_indices.push_back(node_to_index[src]);
    result.dst_indices.push_back(node_to_index[dst]);
  }

  return result;
}

Result<std::shared_ptr<arrow::Table>> GetNodeFeatures(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::string& vertex_type, const std::vector<IdType>& node_ids,
    const std::vector<std::string>& properties, FeatureCache* cache) {
  // Return empty table for empty input
  if (node_ids.empty()) {
    std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
    return arrow::Table::Make(arrow::schema({}), empty_arrays, 0);
  }
  if (properties.empty()) {
    return Status::Invalid("Properties list cannot be empty");
  }

  // Get vertex info
  auto vertex_info = graph_info->GetVertexInfo(vertex_type);
  if (!vertex_info) {
    return Status::Invalid("Vertex type '", vertex_type, "' not found");
  }

  // Validate that all properties exist
  for (const auto& prop : properties) {
    if (!vertex_info->HasProperty(prop)) {
      return Status::Invalid("Property '", prop, "' not found in vertex type '",
                            vertex_type, "'");
    }
  }

  // Group properties by their property group
  std::unordered_map<std::shared_ptr<PropertyGroup>,
                     std::vector<std::string>>
      pg_to_props;
  for (const auto& prop : properties) {
    auto pg = vertex_info->GetPropertyGroup(prop);
    if (!pg) {
      return Status::Invalid("Property group not found for property '", prop,
                            "'");
    }
    pg_to_props[pg].push_back(prop);
  }

  // Build Arrow schema and arrays for result
  std::vector<std::shared_ptr<arrow::Field>> schema_fields;
  std::vector<std::shared_ptr<arrow::Array>> result_arrays;

  const IdType chunk_size = vertex_info->GetChunkSize();

  // For each property group, read the data
  for (const auto& [pg, props] : pg_to_props) {
    auto reader_result =
        VertexPropertyArrowChunkReader::Make(graph_info, vertex_type, pg);
    GAR_RETURN_NOT_OK(reader_result.status());
    auto reader = reader_result.value();

    // node_data[i] = 1-row Table for node_ids[i] (all columns in pg).
    // Pre-populated from cache; remaining entries filled by chunk reads below.
    std::vector<std::shared_ptr<arrow::Table>> node_data(node_ids.size());

    // Step 1: cache lookup per node; collect misses grouped by chunk.
    std::map<IdType, std::vector<size_t>> chunk_to_miss;
    for (size_t i = 0; i < node_ids.size(); i++) {
      if (cache) node_data[i] = cache->Get(graph_info.get(), pg.get(), node_ids[i]);
      if (!node_data[i]) chunk_to_miss[node_ids[i] / chunk_size].push_back(i);
    }

    // Step 2: read each chunk once; bulk-Take all missed rows; cache per node.
    for (const auto& [chunk_id, miss_idxs] : chunk_to_miss) {
      IdType chunk_start = chunk_id * chunk_size;
      GAR_RETURN_NOT_OK(reader->seek(chunk_start));
      auto chunk_res = reader->GetChunk();
      GAR_RETURN_NOT_OK(chunk_res.status());
      auto chunk_tbl = chunk_res.value();

      // Bulk Take: one Arrow call per column for all missed rows in this chunk
      arrow::Int64Builder idx_b;
      auto st = idx_b.Reserve(miss_idxs.size());
      if (!st.ok()) return Status::ArrowError(st.ToString());
      for (size_t i : miss_idxs)
        idx_b.UnsafeAppend(static_cast<int64_t>(node_ids[i] - chunk_start));
      std::shared_ptr<arrow::Array> take_idx;
      st = idx_b.Finish(&take_idx);
      if (!st.ok()) return Status::ArrowError(st.ToString());

      std::vector<std::shared_ptr<arrow::Array>> bulk_cols;
      bulk_cols.reserve(chunk_tbl->num_columns());
      for (int c = 0; c < chunk_tbl->num_columns(); c++) {
        auto res = arrow::compute::Take(chunk_tbl->column(c)->chunk(0), take_idx);
        if (!res.ok()) return Status::ArrowError(res.status().ToString());
        bulk_cols.push_back(res.ValueOrDie().make_array());
      }

      // Split bulk result into per-node 1-row tables and cache each
      auto row_schema = chunk_tbl->schema();
      for (size_t li = 0; li < miss_idxs.size(); li++) {
        arrow::Int64Builder rb;
        st = rb.Append(static_cast<int64_t>(li));
        if (!st.ok()) return Status::ArrowError(st.ToString());
        std::shared_ptr<arrow::Array> row_take;
        st = rb.Finish(&row_take);
        if (!st.ok()) return Status::ArrowError(st.ToString());

        std::vector<std::shared_ptr<arrow::Array>> row_arrs;
        row_arrs.reserve(bulk_cols.size());
        for (const auto& col : bulk_cols) {
          auto res = arrow::compute::Take(col, row_take);
          if (!res.ok()) return Status::ArrowError(res.status().ToString());
          row_arrs.push_back(res.ValueOrDie().make_array());
        }
        auto row_tbl = arrow::Table::Make(row_schema, row_arrs, 1);

        size_t orig = miss_idxs[li];
        if (cache) cache->Put(graph_info.get(), pg.get(), node_ids[orig], row_tbl);
        node_data[orig] = row_tbl;
      }
    }

    // Step 3: assemble output columns from per-node rows in node_ids order
    for (const auto& prop : props) {
      std::vector<std::shared_ptr<arrow::Array>> col_chunks;
      col_chunks.reserve(node_ids.size());
      for (const auto& row : node_data) {
        auto col = row->GetColumnByName(prop);
        if (!col) return Status::Invalid("Column '", prop, "' not found in row");
        col_chunks.push_back(col->chunk(0));
      }
      auto concat = arrow::Concatenate(col_chunks);
      if (!concat.ok()) return Status::ArrowError(concat.status().ToString());

      auto prop_type_result = vertex_info->GetPropertyType(prop);
      GAR_RETURN_NOT_OK(prop_type_result.status());
      auto arrow_type =
          DataType::DataTypeToArrowDataType(prop_type_result.value());

      schema_fields.push_back(arrow::field(prop, arrow_type));
      result_arrays.push_back(concat.ValueOrDie());
    }
  }

  // Create final table
  auto schema = arrow::schema(schema_fields);
  return arrow::Table::Make(schema, result_arrays, node_ids.size());
}

}  // namespace graphar::ml
