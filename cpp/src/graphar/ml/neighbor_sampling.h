#pragma once

#include <memory>
#include <string>
#include <vector>

#include "graphar/fwd.h"

namespace arrow {
class Table;
}

namespace graphar::ml {

/**
 * Result of neighbor sampling operation.
 */
struct SamplingResult {
  std::vector<IdType> sampled_nodes;  // Unique internal IDs in sampling order [N]
  std::vector<IdType> src_indices;    // Edge sources as indices [E]
  std::vector<IdType> dst_indices;    // Edge dests as indices [E]
};

/**
 * Sample multi-hop neighbors for given seed nodes.
 *
 * @param graph_info The graph info containing metadata
 * @param vertex_type The vertex type to sample from
 * @param edge_type The edge type to traverse
 * @param seed_nodes Vector of seed node internal IDs
 * @param fanout Vector of neighbor counts per hop (e.g., [10, 5] for 2-hop)
 * @return Result containing sampled nodes and edge structure
 */
Result<SamplingResult> SampleNeighbors(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::string& vertex_type, const std::string& edge_type,
    const std::vector<IdType>& seed_nodes, const std::vector<int>& fanout,
    uint64_t seed);

/**
 * Fetch node properties for given internal IDs.
 *
 * @param graph_info The graph info containing metadata
 * @param vertex_type The vertex type
 * @param node_ids Vector of node internal IDs
 * @param properties Vector of property names to fetch
 * @return Result containing Arrow Table with requested properties
 */
Result<std::shared_ptr<arrow::Table>> GetNodeFeatures(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::string& vertex_type, const std::vector<IdType>& node_ids,
    const std::vector<std::string>& properties);

}  // namespace graphar::ml
