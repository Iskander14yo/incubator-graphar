#include "graphar/ml/neighbor_sampling.h"

#include <algorithm>
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "arrow/api.h"
#include "graphar/arrow/chunk_reader.h"
#include "graphar/graph_info.h"
#include "graphar/high-level/graph_reader.h"

namespace graphar::ml {

Result<SamplingResult> SampleNeighbors(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::string& vertex_type, const std::string& edge_type,
    const std::vector<IdType>& seed_nodes, const std::vector<int>& fanout,
    uint64_t seed) {
  // Validate inputs
  if (seed_nodes.empty()) {
    return SamplingResult{};
  }
  if (fanout.empty()) {
    return Status::Invalid("Fanout cannot be empty");
  }

  // Get edge info for the specified types
  auto edge_info = graph_info->GetEdgeInfo(vertex_type, edge_type, vertex_type);
  if (!edge_info) {
    return Status::Invalid("Edge type '", edge_type, "' not found for vertex type '",
                          vertex_type, "'");
  }

  // Check if CSR layout is available
  if (!edge_info->HasAdjacentListType(AdjListType::ordered_by_source)) {
    return Status::Invalid(
        "CSR layout (ordered_by_source) not available for edge type '", edge_type,
        "'");
  }

  // Create EdgesCollection with CSR layout
  auto edges_collection_result = EdgesCollection::Make(
      graph_info, vertex_type, edge_type, vertex_type,
      AdjListType::ordered_by_source);
  GAR_RETURN_NOT_OK(edges_collection_result.status());
  auto edges_collection = edges_collection_result.value();

  // Track all sampled nodes and edges
  std::unordered_set<IdType> all_sampled_nodes_set(seed_nodes.begin(),
                                                     seed_nodes.end());
  std::vector<std::pair<IdType, IdType>> edge_list;  // (src, dst) pairs

  // Current frontier starts with seed nodes
  std::vector<IdType> current_frontier = seed_nodes;

  // Random number generator for deterministic sampling
  std::mt19937 gen(seed);

  // Process each hop
  for (size_t hop = 0; hop < fanout.size(); ++hop) {
    int max_neighbors = fanout[hop];
    std::vector<IdType> next_frontier;

    // For each node in current frontier
    for (IdType src_node : current_frontier) {
      // Find edges starting from this node
      auto edge_iter = edges_collection->find_src(src_node, edges_collection->begin());
      
      // Collect all neighbors
      std::vector<IdType> neighbors;
      while (edge_iter != edges_collection->end() && 
             edge_iter.source() == src_node) {
        IdType dst_node = edge_iter.destination();
        neighbors.push_back(dst_node);
        ++edge_iter;
      }

      // Sample up to max_neighbors
      if (neighbors.size() > static_cast<size_t>(max_neighbors)) {
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
