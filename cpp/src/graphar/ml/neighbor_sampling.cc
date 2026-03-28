#include "graphar/ml/neighbor_sampling.h"

#include <algorithm>
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "arrow/api.h"
#include "graphar/arrow/chunk_reader.h"
#include "graphar/graph_info.h"
#include "graphar/result.h"
#include "graphar/types.h"

namespace graphar::ml {

Result<SamplingResult> SampleNeighbors(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::string& vertex_type, const std::string& edge_type,
    const std::vector<IdType>& seed_nodes, const std::vector<int>& fanout,
    uint64_t seed) {
  if (seed_nodes.empty()) return SamplingResult{};
  if (fanout.empty()) return Status::Invalid("Fanout cannot be empty");

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
  const auto adj_type = AdjListType::ordered_by_source;
  const IdType src_chunk_size = edge_info->GetSrcChunkSize();

  // Persistent readers — chunk caches survive across seeks within the same chunk,
  // unlike EdgesCollection/EdgeIter which re-create readers per source node.
  AdjListOffsetArrowChunkReader offset_reader(edge_info, adj_type, prefix);
  AdjListArrowChunkReader adj_reader(edge_info, adj_type, prefix);

  std::unordered_set<IdType> all_sampled_nodes_set(seed_nodes.begin(),
                                                   seed_nodes.end());
  std::vector<std::pair<IdType, IdType>> edge_list;
  std::vector<IdType> current_frontier = seed_nodes;
  std::mt19937 gen(seed);

  for (size_t hop = 0; hop < fanout.size(); ++hop) {
    int max_neighbors = fanout[hop];
    std::vector<IdType> next_frontier;

    // Sort frontier for sequential chunk access — maximizes Parquet cache hits
    std::vector<IdType> sorted_frontier = current_frontier;
    std::sort(sorted_frontier.begin(), sorted_frontier.end());

    IdType prev_vertex_chunk = -1;

    for (IdType src_node : sorted_frontier) {
      IdType vertex_chunk_idx = src_node / src_chunk_size;

      if (vertex_chunk_idx != prev_vertex_chunk) {
        GAR_RETURN_NOT_OK(adj_reader.seek_chunk_index(vertex_chunk_idx));
        prev_vertex_chunk = vertex_chunk_idx;
      }

      // Read edge range from CSR offset array (cached per vertex chunk)
      GAR_RETURN_NOT_OK(offset_reader.seek(src_node));
      GAR_ASSIGN_OR_RAISE(auto offset_arr, offset_reader.GetChunk());
      auto offsets = std::static_pointer_cast<arrow::Int64Array>(offset_arr);
      if (offsets->length() < 2) continue;

      IdType begin_off = offsets->Value(0);
      IdType end_off = offsets->Value(1);
      if (begin_off >= end_off) continue;

      // Extract destination IDs directly from Arrow arrays — no per-edge
      // seek/GetChunk/Slice like EdgeIter::destination() does.
      std::vector<IdType> neighbors;
      IdType total_edges = end_off - begin_off;
      neighbors.reserve(total_edges);

      IdType remaining = total_edges;
      IdType cur_off = begin_off;

      while (remaining > 0) {
        GAR_RETURN_NOT_OK(adj_reader.seek(cur_off));
        GAR_ASSIGN_OR_RAISE(auto chunk_table, adj_reader.GetChunk());
        if (!chunk_table) break;

        auto dst_col = std::static_pointer_cast<arrow::Int64Array>(
            chunk_table->column(1)->chunk(0));
        IdType batch = std::min(remaining, static_cast<IdType>(dst_col->length()));
        const int64_t* raw = dst_col->raw_values();
        neighbors.insert(neighbors.end(), raw, raw + batch);
        remaining -= batch;
        cur_off += batch;
      }

      if (static_cast<int>(neighbors.size()) > max_neighbors) {
        std::shuffle(neighbors.begin(), neighbors.end(), gen);
        neighbors.resize(max_neighbors);
      }

      // Add sampled neighbors to results
      for (IdType dst_node : neighbors) {
        edge_list.emplace_back(src_node, dst_node);
        
        // Add to sampled nodes set and next frontier
        if (all_sampled_nodes_set.insert(dst_node).second) {
          next_frontier.push_back(dst_node);
        }
      }
    }

    current_frontier = std::move(next_frontier);
  }

  // Convert to result format
  SamplingResult result;
  
  // Create ordered list of sampled nodes
  result.sampled_nodes.assign(all_sampled_nodes_set.begin(),
                              all_sampled_nodes_set.end());
  std::sort(result.sampled_nodes.begin(), result.sampled_nodes.end());

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
    const std::vector<std::string>& properties) {
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

  // For each property group, read the data
  for (const auto& [pg, props] : pg_to_props) {
    // Create reader for this property group
    auto reader_result =
        VertexPropertyArrowChunkReader::Make(graph_info, vertex_type, pg);
    GAR_RETURN_NOT_OK(reader_result.status());
    auto reader = reader_result.value();

    // Group node_ids by chunk to minimize chunk reads
    IdType chunk_size = vertex_info->GetChunkSize();
    std::map<IdType, std::vector<size_t>> chunk_to_indices;
    for (size_t i = 0; i < node_ids.size(); i++) {
      IdType chunk_id = node_ids[i] / chunk_size;
      chunk_to_indices[chunk_id].push_back(i);
    }

    // Build arrays for each property in this group
    std::unordered_map<std::string, std::vector<std::shared_ptr<arrow::Scalar>>>
        prop_scalars;
    for (const auto& prop : props) {
      prop_scalars[prop].resize(node_ids.size());
    }

    // Read each chunk and extract needed rows
    for (const auto& [chunk_id, indices] : chunk_to_indices) {
      // Seek to the beginning of this chunk
      IdType chunk_start_node = chunk_id * chunk_size;
      GAR_RETURN_NOT_OK(reader->seek(chunk_start_node));

      // Read the chunk
      auto chunk_result = reader->GetChunk();
      GAR_RETURN_NOT_OK(chunk_result.status());
      auto chunk_table = chunk_result.value();

      // Extract values for each node_id in this chunk
      for (size_t idx : indices) {
        IdType node_id = node_ids[idx];
        IdType row_in_chunk = node_id - chunk_start_node;

        // Extract value for each property
        for (const auto& prop : props) {
          auto column = chunk_table->GetColumnByName(prop);
          if (!column) {
            return Status::Invalid("Column '", prop, "' not found in chunk");
          }
          auto scalar_result = column->chunk(0)->GetScalar(row_in_chunk);
          if (!scalar_result.ok()) {
            return Status::ArrowError(scalar_result.status().ToString());
          }
          prop_scalars[prop][idx] = scalar_result.ValueOrDie();
        }
      }
    }

    // Convert scalars to arrays for each property
    for (const auto& prop : props) {
      auto prop_type_result = vertex_info->GetPropertyType(prop);
      GAR_RETURN_NOT_OK(prop_type_result.status());
      auto prop_type = prop_type_result.value();

      // Build arrow array from scalars
      auto arrow_type = DataType::DataTypeToArrowDataType(prop_type);
      auto builder_result = arrow::MakeBuilder(arrow_type, arrow::default_memory_pool());
      if (!builder_result.ok()) {
        return Status::ArrowError(builder_result.status().ToString());
      }
      auto builder = std::move(builder_result).ValueOrDie();

      for (const auto& scalar : prop_scalars[prop]) {
        auto append_status = builder->AppendScalar(*scalar);
        if (!append_status.ok()) {
          return Status::ArrowError(append_status.ToString());
        }
      }

      std::shared_ptr<arrow::Array> array;
      auto finish_status = builder->Finish(&array);
      if (!finish_status.ok()) {
        return Status::ArrowError(finish_status.ToString());
      }

      schema_fields.push_back(arrow::field(prop, arrow_type));
      result_arrays.push_back(array);
    }
  }

  // Create final table
  auto schema = arrow::schema(schema_fields);
  return arrow::Table::Make(schema, result_arrays, node_ids.size());
}

}  // namespace graphar::ml
