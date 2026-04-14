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

using PropertyGroupMap =
    std::unordered_map<std::shared_ptr<PropertyGroup>,
                       std::vector<std::string>>;

size_t ArrayDataBytes(const std::shared_ptr<arrow::ArrayData>& data) {
  if (data == nullptr) return 0;
  size_t total = 0;
  for (const auto& buffer : data->buffers) {
    if (buffer != nullptr) total += static_cast<size_t>(buffer->size());
  }
  for (const auto& child : data->child_data) {
    total += ArrayDataBytes(child);
  }
  total += ArrayDataBytes(data->dictionary);
  return total;
}

size_t ScalarBytes(const std::shared_ptr<arrow::Scalar>& scalar) {
  if (scalar == nullptr || !scalar->is_valid) return 1;

  switch (scalar->type->id()) {
    case arrow::Type::BINARY:
    case arrow::Type::STRING:
    case arrow::Type::LARGE_BINARY:
    case arrow::Type::LARGE_STRING:
    case arrow::Type::BINARY_VIEW:
    case arrow::Type::STRING_VIEW: {
      auto* binary = static_cast<const arrow::BaseBinaryScalar*>(scalar.get());
      return binary->value == nullptr ? 0
                                      : static_cast<size_t>(binary->value->size());
    }
    case arrow::Type::LIST:
    case arrow::Type::LARGE_LIST:
    case arrow::Type::LIST_VIEW:
    case arrow::Type::LARGE_LIST_VIEW:
    case arrow::Type::MAP:
    case arrow::Type::FIXED_SIZE_LIST: {
      auto* list = static_cast<const arrow::BaseListScalar*>(scalar.get());
      return list->value == nullptr ? 0 : ArrayDataBytes(list->value->data());
    }
    case arrow::Type::STRUCT: {
      auto* struct_scalar = static_cast<const arrow::StructScalar*>(scalar.get());
      size_t total = 0;
      for (const auto& child : struct_scalar->value) {
        total += ScalarBytes(child);
      }
      return total;
    }
    case arrow::Type::SPARSE_UNION:
    case arrow::Type::DENSE_UNION: {
      auto* union_scalar = static_cast<const arrow::UnionScalar*>(scalar.get());
      return ScalarBytes(union_scalar->child_value());
    }
    case arrow::Type::DICTIONARY: {
      auto* dict_scalar = static_cast<const arrow::DictionaryScalar*>(scalar.get());
      return ScalarBytes(dict_scalar->value.index) +
             ArrayDataBytes(dict_scalar->value.dictionary->data());
    }
    case arrow::Type::EXTENSION: {
      auto* ext_scalar = static_cast<const arrow::ExtensionScalar*>(scalar.get());
      return ScalarBytes(ext_scalar->value);
    }
    default: {
      auto* primitive =
          dynamic_cast<const arrow::internal::PrimitiveScalarBase*>(scalar.get());
      return primitive == nullptr ? sizeof(*scalar) : primitive->view().size();
    }
  }
}

Result<std::shared_ptr<arrow::Table>> GetNodeFeaturesUncached(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::shared_ptr<VertexInfo>& vertex_info,
    const std::string& vertex_type, const std::vector<IdType>& node_ids,
    const PropertyGroupMap& pg_to_props) {
  std::vector<std::shared_ptr<arrow::Field>> schema_fields;
  std::vector<std::shared_ptr<arrow::Array>> result_arrays;

  for (const auto& [pg, props] : pg_to_props) {
    auto reader_result =
        VertexPropertyArrowChunkReader::Make(graph_info, vertex_type, pg);
    GAR_RETURN_NOT_OK(reader_result.status());
    auto reader = reader_result.value();

    IdType chunk_size = vertex_info->GetChunkSize();
    std::map<IdType, std::vector<size_t>> chunk_to_indices;
    for (size_t i = 0; i < node_ids.size(); ++i) {
      chunk_to_indices[node_ids[i] / chunk_size].push_back(i);
    }

    std::unordered_map<std::string, std::vector<std::shared_ptr<arrow::Array>>>
        prop_taken;
    std::vector<int64_t> concat_to_output;
    concat_to_output.reserve(node_ids.size());

    for (const auto& [chunk_id, indices] : chunk_to_indices) {
      IdType chunk_start = chunk_id * chunk_size;
      GAR_RETURN_NOT_OK(reader->seek(chunk_start));
      auto chunk_result = reader->GetChunk();
      GAR_RETURN_NOT_OK(chunk_result.status());
      auto chunk_table = chunk_result.value();

      arrow::Int64Builder idx_builder;
      auto st = idx_builder.Reserve(indices.size());
      if (!st.ok()) return Status::ArrowError(st.ToString());
      for (size_t idx : indices) {
        idx_builder.UnsafeAppend(node_ids[idx] - chunk_start);
      }
      std::shared_ptr<arrow::Array> take_indices;
      st = idx_builder.Finish(&take_indices);
      if (!st.ok()) return Status::ArrowError(st.ToString());

      for (const auto& prop : props) {
        auto column = chunk_table->GetColumnByName(prop);
        if (!column) {
          return Status::Invalid("Column '", prop, "' not found in chunk");
        }
        auto take_result =
            arrow::compute::Take(column->chunk(0), take_indices);
        if (!take_result.ok()) {
          return Status::ArrowError(take_result.status().ToString());
        }
        prop_taken[prop].push_back(take_result.ValueOrDie().make_array());
      }

      for (size_t idx : indices) {
        concat_to_output.push_back(static_cast<int64_t>(idx));
      }
    }

    std::vector<int64_t> inv_perm(node_ids.size());
    for (size_t i = 0; i < concat_to_output.size(); ++i) {
      inv_perm[concat_to_output[i]] = static_cast<int64_t>(i);
    }
    arrow::Int64Builder perm_builder;
    auto st = perm_builder.AppendValues(inv_perm);
    if (!st.ok()) return Status::ArrowError(st.ToString());
    std::shared_ptr<arrow::Array> perm_array;
    st = perm_builder.Finish(&perm_array);
    if (!st.ok()) return Status::ArrowError(st.ToString());

    for (const auto& prop : props) {
      auto concat_result = arrow::Concatenate(prop_taken[prop]);
      if (!concat_result.ok()) {
        return Status::ArrowError(concat_result.status().ToString());
      }
      auto reorder_result =
          arrow::compute::Take(concat_result.ValueOrDie(), perm_array);
      if (!reorder_result.ok()) {
        return Status::ArrowError(reorder_result.status().ToString());
      }

      auto prop_type_result = vertex_info->GetPropertyType(prop);
      GAR_RETURN_NOT_OK(prop_type_result.status());
      schema_fields.push_back(arrow::field(
          prop, DataType::DataTypeToArrowDataType(prop_type_result.value())));
      result_arrays.push_back(reorder_result.ValueOrDie().make_array());
    }
  }

  return arrow::Table::Make(arrow::schema(schema_fields), result_arrays,
                            node_ids.size());
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
  PropertyGroupMap pg_to_props;
  for (const auto& prop : properties) {
    auto pg = vertex_info->GetPropertyGroup(prop);
    if (!pg) {
      return Status::Invalid("Property group not found for property '", prop,
                            "'");
    }
    pg_to_props[pg].push_back(prop);
  }

  if (cache == nullptr) {
    return GetNodeFeaturesUncached(graph_info, vertex_info, vertex_type,
                                   node_ids, pg_to_props);
  }

  std::vector<std::shared_ptr<arrow::Field>> schema_fields;
  std::vector<std::shared_ptr<arrow::Array>> result_arrays;
  const IdType chunk_size = vertex_info->GetChunkSize();

  for (const auto& [pg, props] : pg_to_props) {
    auto reader_result =
        VertexPropertyArrowChunkReader::Make(graph_info, vertex_type, pg);
    GAR_RETURN_NOT_OK(reader_result.status());
    auto reader = reader_result.value();

    std::unordered_map<std::string, int> prop_to_index;
    const auto& pg_props = pg->GetProperties();
    for (size_t i = 0; i < pg_props.size(); ++i) {
      prop_to_index[pg_props[i].name] = static_cast<int>(i);
    }

    std::vector<int> requested_indices;
    requested_indices.reserve(props.size());
    std::vector<std::unique_ptr<arrow::ArrayBuilder>> builders;
    builders.reserve(props.size());
    for (const auto& prop : props) {
      auto prop_it = prop_to_index.find(prop);
      if (prop_it == prop_to_index.end()) {
        return Status::Invalid("Property '", prop,
                               "' not found in property group");
      }
      requested_indices.push_back(prop_it->second);

      auto prop_type_result = vertex_info->GetPropertyType(prop);
      GAR_RETURN_NOT_OK(prop_type_result.status());
      auto arrow_type =
          DataType::DataTypeToArrowDataType(prop_type_result.value());
      schema_fields.push_back(arrow::field(prop, arrow_type));

      auto builder_result =
          arrow::MakeBuilder(arrow_type, arrow::default_memory_pool());
      if (!builder_result.ok()) {
        return Status::ArrowError(builder_result.status().ToString());
      }
      auto builder = std::move(builder_result).ValueOrDie();
      auto st = builder->Reserve(node_ids.size());
      if (!st.ok()) return Status::ArrowError(st.ToString());
      builders.push_back(std::move(builder));
    }

    std::vector<std::shared_ptr<const FeatureCache::CachedRow>> node_rows(
        node_ids.size());
    std::map<IdType, std::vector<size_t>> chunk_to_miss;
    for (size_t i = 0; i < node_ids.size(); ++i) {
      node_rows[i] = cache->Get(graph_info.get(), pg.get(), node_ids[i]);
      if (node_rows[i] == nullptr) {
        chunk_to_miss[node_ids[i] / chunk_size].push_back(i);
      }
    }

    std::unordered_set<IdType> all_chunk_ids;
    for (auto id : node_ids) all_chunk_ids.insert(id / chunk_size);
    cache->RecordChunksRead(chunk_to_miss.size());
    cache->RecordChunksSkipped(all_chunk_ids.size() - chunk_to_miss.size());

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
      bulk_cols.reserve(pg_props.size());
      for (const auto& pg_prop : pg_props) {
        auto column = chunk_tbl->GetColumnByName(pg_prop.name);
        if (!column) {
          return Status::Invalid("Column '", pg_prop.name,
                                 "' not found in chunk");
        }
        auto res = arrow::compute::Take(column->chunk(0), take_idx);
        if (!res.ok()) return Status::ArrowError(res.status().ToString());
        bulk_cols.push_back(res.ValueOrDie().make_array());
      }

      for (size_t li = 0; li < miss_idxs.size(); ++li) {
        auto row = std::make_shared<FeatureCache::CachedRow>();
        row->values.reserve(bulk_cols.size());
        for (const auto& col : bulk_cols) {
          auto scalar_result = col->GetScalar(li);
          if (!scalar_result.ok()) {
            return Status::ArrowError(scalar_result.status().ToString());
          }
          auto scalar = scalar_result.ValueOrDie();
          row->size_bytes += ScalarBytes(scalar);
          row->values.push_back(std::move(scalar));
        }

        size_t orig = miss_idxs[li];
        cache->Put(graph_info.get(), pg.get(), node_ids[orig], row);
        node_rows[orig] = row;
      }
    }

    for (const auto& row : node_rows) {
      if (row == nullptr) {
        return Status::Invalid("Missing cached row while assembling result");
      }
      for (size_t i = 0; i < requested_indices.size(); ++i) {
        const auto& scalar = row->values[requested_indices[i]];
        auto st = builders[i]->AppendScalar(*scalar);
        if (!st.ok()) return Status::ArrowError(st.ToString());
      }
    }

    for (auto& builder : builders) {
      std::shared_ptr<arrow::Array> array;
      auto st = builder->Finish(&array);
      if (!st.ok()) return Status::ArrowError(st.ToString());
      result_arrays.push_back(std::move(array));
    }
  }

  return arrow::Table::Make(arrow::schema(schema_fields), result_arrays,
                            node_ids.size());
}

}  // namespace graphar::ml
