#include "graphar/ml/static_feature_cache.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <unordered_map>

#include "arrow/api.h"
#include "arrow/compute/api.h"
#include "graphar/graph_info.h"
#include "graphar/ml/arrow_size_util.h"
#include "graphar/ml/get_node_features_uncached.h"
#include "graphar/types.h"

namespace graphar::ml {
namespace {

size_t ChunkedBytes(const std::shared_ptr<arrow::ChunkedArray>& col) {
  if (!col) return 0;
  size_t t = 0;
  for (int i = 0; i < col->num_chunks(); ++i) {
    t += ArrayDataBytes(col->chunk(i)->data());
  }
  return t;
}

std::pair<double, std::string> EstimatePinBytes(
    const std::shared_ptr<VertexInfo>& vertex_info,
    const std::vector<std::string>& properties, size_t num_nodes) {
  double est = 0;
  bool any_var = false;
  for (const auto& prop : properties) {
    auto prop_type_result = vertex_info->GetPropertyType(prop);
    if (!prop_type_result.status().ok()) continue;
    auto arrow_type =
        DataType::DataTypeToArrowDataType(prop_type_result.value());
    if (arrow::is_fixed_width(*arrow_type)) {
      auto fw = std::static_pointer_cast<arrow::FixedWidthType>(arrow_type);
      est += static_cast<double>(num_nodes) *
             static_cast<double>((fw->bit_width() + 7) / 8);
    } else {
      any_var = true;
    }
  }
  std::string note =
      any_var ? "estimate excludes variable-width properties" : "fixed-width only";
  return {est, note};
}

}  // namespace

StaticFeatureCache::StaticFeatureCache(std::shared_ptr<GraphInfo> graph_info)
    : graph_info_(std::move(graph_info)) {}

Status StaticFeatureCache::Pin(const std::string& vertex_type,
                               const std::vector<IdType>& node_ids,
                               const std::vector<std::string>& properties) {
  if (properties.empty()) {
    return Status::Invalid("Properties list cannot be empty");
  }
  if (node_ids.empty()) {
    return Status::OK();
  }

  auto vertex_info = graph_info_->GetVertexInfo(vertex_type);
  if (!vertex_info) {
    return Status::Invalid("Vertex type '", vertex_type, "' not found");
  }

  for (const auto& prop : properties) {
    if (!vertex_info->HasProperty(prop)) {
      return Status::Invalid("Property '", prop, "' not found in vertex type '",
                             vertex_type, "'");
    }
  }

  PropertyGroupMap pg_to_props;
  for (const auto& prop : properties) {
    auto pg = vertex_info->GetPropertyGroup(prop);
    if (!pg) {
      return Status::Invalid("Property group not found for property '", prop,
                             "'");
    }
    pg_to_props[pg].push_back(prop);
  }

  auto [est_bytes, est_note] = EstimatePinBytes(vertex_info, properties,
                                                node_ids.size());
  fprintf(stderr,
          "[graphar::ml::StaticFeatureCache] Pinning %zu nodes, estimated %.1f "
          "MB (%s)\n",
          node_ids.size(), est_bytes / 1e6, est_note.c_str());

  std::unique_lock<std::shared_mutex> lock(mu_);

  for (const auto& [pg, props_requested] : pg_to_props) {
    auto& pin = pins_[vertex_type][pg];

    std::vector<std::string> union_props = pin.props_order;
    for (const auto& p : props_requested) {
      if (std::find(union_props.begin(), union_props.end(), p) ==
          union_props.end()) {
        union_props.push_back(p);
      }
    }

    std::unordered_set<IdType> have(pin.rowid_order.begin(),
                                    pin.rowid_order.end());
    std::vector<IdType> union_ids = pin.rowid_order;
    for (IdType id : node_ids) {
      if (!have.count(id)) {
        union_ids.push_back(id);
        have.insert(id);
      }
    }

    if (union_ids == pin.rowid_order && union_props == pin.props_order &&
        !pin.props_order.empty()) {
      continue;
    }

    PropertyGroupMap one;
    one[pg] = union_props;
    auto table_result = GetNodeFeaturesUncached(graph_info_, vertex_info,
                                                vertex_type, union_ids, one);
    GAR_RETURN_NOT_OK(table_result.status());
    auto table = table_result.value();

    pin.props_order = std::move(union_props);
    pin.rowid_order = std::move(union_ids);
    pin.id_to_row.clear();
    for (size_t i = 0; i < pin.rowid_order.size(); ++i) {
      pin.id_to_row[pin.rowid_order[i]] = static_cast<int64_t>(i);
    }
    pin.columns.clear();
    for (const auto& name : pin.props_order) {
      pin.columns[name] = table->GetColumnByName(name);
    }

    for (IdType id : pin.rowid_order) {
      pinned_ids_.insert(id);
    }
  }

  size_t actual = ComputeStorageBytesUnlocked();
  fprintf(stderr,
          "[graphar::ml::StaticFeatureCache] Pinned actual %.1f MB\n",
          static_cast<double>(actual) / 1e6);

  return Status::OK();
}

size_t StaticFeatureCache::ComputeStorageBytesUnlocked() const {
  size_t t = 0;
  for (const auto& [vt, inner] : pins_) {
    static_cast<void>(vt);
    for (const auto& [pg, pin] : inner) {
      static_cast<void>(pg);
      for (const auto& [name, col] : pin.columns) {
        static_cast<void>(name);
        t += ChunkedBytes(col);
      }
    }
  }
  return t;
}

size_t StaticFeatureCache::ComputeStorageBytes() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return ComputeStorageBytesUnlocked();
}

Result<StaticFeatureCache::LookupResult> StaticFeatureCache::Lookup(
    const std::string& vertex_type, const std::vector<IdType>& node_ids,
    const std::vector<std::string>& properties) {
  LookupResult out;
  if (node_ids.empty()) {
    return out;
  }
  if (properties.empty()) {
    return Status::Invalid("Properties list cannot be empty");
  }

  auto vertex_info = graph_info_->GetVertexInfo(vertex_type);
  if (!vertex_info) {
    return Status::Invalid("Vertex type '", vertex_type, "' not found");
  }

  PropertyGroupMap pg_to_props;
  for (const auto& prop : properties) {
    if (!vertex_info->HasProperty(prop)) {
      return Status::Invalid("Property '", prop, "' not found in vertex type '",
                             vertex_type, "'");
    }
    auto pg = vertex_info->GetPropertyGroup(prop);
    if (!pg) {
      return Status::Invalid("Property group not found for property '", prop,
                             "'");
    }
    pg_to_props[pg].push_back(prop);
  }

  std::shared_lock<std::shared_mutex> lock(mu_);

  auto vt_it = pins_.find(vertex_type);
  std::vector<size_t> hit_pos;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    IdType nid = node_ids[i];
    bool row_hit = true;
    if (vt_it == pins_.end()) {
      row_hit = false;
    } else {
      for (const auto& [pg, props] : pg_to_props) {
        auto pg_it = vt_it->second.find(pg);
        if (pg_it == vt_it->second.end()) {
          row_hit = false;
          break;
        }
        const PgPin& pin = pg_it->second;
        if (pin.id_to_row.find(nid) == pin.id_to_row.end()) {
          row_hit = false;
          break;
        }
        for (const auto& prop : props) {
          if (pin.columns.find(prop) == pin.columns.end()) {
            row_hit = false;
            break;
          }
        }
        if (!row_hit) break;
      }
    }
    if (row_hit) {
      hit_pos.push_back(i);
    } else {
      out.miss_positions.push_back(i);
      out.miss_node_ids.push_back(nid);
    }
  }

  hits_.fetch_add(hit_pos.size());
  misses_.fetch_add(out.miss_positions.size());

  out.hit_positions = hit_pos;

  if (hit_pos.empty()) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrs;
    for (const auto& prop : properties) {
      auto prop_type_result = vertex_info->GetPropertyType(prop);
      GAR_RETURN_NOT_OK(prop_type_result.status());
      auto at = DataType::DataTypeToArrowDataType(prop_type_result.value());
      fields.push_back(arrow::field(prop, at));
      auto empty_st = arrow::MakeEmptyArray(at);
      if (!empty_st.ok()) {
        return Status::ArrowError(empty_st.status().ToString());
      }
      arrs.push_back(empty_st.ValueOrDie());
    }
    out.hits = arrow::Table::Make(arrow::schema(fields), arrs, 0);
    return out;
  }

  if (vt_it == pins_.end()) {
    return Status::Invalid("Lookup internal state: hits without pinned data");
  }

  std::vector<std::shared_ptr<arrow::Field>> schema_fields;
  std::vector<std::shared_ptr<arrow::Array>> out_arrays;
  schema_fields.reserve(properties.size());
  out_arrays.reserve(properties.size());

  for (const auto& [pg, props] : pg_to_props) {
    const PgPin& pin = vt_it->second.at(pg);
    arrow::Int64Builder idx_b;
    auto st = idx_b.Reserve(static_cast<int64_t>(hit_pos.size()));
    if (!st.ok()) return Status::ArrowError(st.ToString());
    for (size_t hi = 0; hi < hit_pos.size(); ++hi) {
      IdType nid = node_ids[hit_pos[hi]];
      idx_b.UnsafeAppend(pin.id_to_row.at(nid));
    }
    std::shared_ptr<arrow::Array> idx_arr;
    st = idx_b.Finish(&idx_arr);
    if (!st.ok()) return Status::ArrowError(st.ToString());

    for (const auto& prop : props) {
      auto prop_type_result = vertex_info->GetPropertyType(prop);
      GAR_RETURN_NOT_OK(prop_type_result.status());
      schema_fields.push_back(arrow::field(
          prop, DataType::DataTypeToArrowDataType(prop_type_result.value())));

      auto take_result =
          arrow::compute::Take(arrow::Datum(pin.columns.at(prop)),
                               arrow::Datum(idx_arr));
      if (!take_result.ok()) {
        return Status::ArrowError(take_result.status().ToString());
      }
      auto taken = take_result.ValueOrDie().chunked_array();
      std::shared_ptr<arrow::Array> out_col;
      if (taken->num_chunks() == 1) {
        out_col = taken->chunk(0);
      } else {
        std::vector<std::shared_ptr<arrow::Array>> chs;
        for (int c = 0; c < taken->num_chunks(); ++c) {
          chs.push_back(taken->chunk(c));
        }
        auto concat = arrow::Concatenate(chs);
        if (!concat.ok()) {
          return Status::ArrowError(concat.status().ToString());
        }
        out_col = concat.ValueOrDie();
      }
      out_arrays.push_back(std::move(out_col));
    }
  }

  out.hits = arrow::Table::Make(arrow::schema(schema_fields), out_arrays,
                                static_cast<int64_t>(hit_pos.size()));
  return out;
}

size_t StaticFeatureCache::num_nodes() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return pinned_ids_.size();
}

size_t StaticFeatureCache::size_bytes() const { return ComputeStorageBytes(); }

size_t StaticFeatureCache::hits() const { return hits_.load(); }

size_t StaticFeatureCache::misses() const { return misses_.load(); }

double StaticFeatureCache::hit_rate() const {
  size_t h = hits_.load();
  size_t m = misses_.load();
  if (h + m == 0) return 0.0;
  return static_cast<double>(h) / static_cast<double>(h + m);
}

}  // namespace ml
