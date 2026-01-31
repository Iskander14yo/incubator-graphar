#include "graphar/ml/neighbor_sampling.h"

#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "arrow/api.h"
#include "graphar/graph_info.h"
#include "graphar/high-level/graph_reader.h"

namespace graphar::ml {

Result<SamplingResult> SampleNeighbors(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::string& vertex_type, const std::string& edge_type,
    const std::vector<IdType>& seed_nodes, const std::vector<int>& fanout) {
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

  // Random number generator for sampling
  std::random_device rd;
  std::mt19937 gen(rd());

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

  // TODO: Implement feature retrieval
  return Status::Invalid("GetNodeFeatures not yet implemented");
}

}  // namespace graphar::ml
