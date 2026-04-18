#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "graphar/graph_info.h"

namespace arrow {
class Table;
}

namespace graphar::ml {

using PropertyGroupMap =
    std::unordered_map<std::shared_ptr<PropertyGroup>, std::vector<std::string>>;

Result<std::shared_ptr<arrow::Table>> GetNodeFeaturesUncached(
    const std::shared_ptr<GraphInfo>& graph_info,
    const std::shared_ptr<VertexInfo>& vertex_info,
    const std::string& vertex_type, const std::vector<IdType>& node_ids,
    const PropertyGroupMap& pg_to_props);

}  // namespace graphar::ml
