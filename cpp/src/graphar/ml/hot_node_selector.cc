#include "graphar/ml/hot_node_selector.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "arrow/api.h"
#include "graphar/arrow/chunk_reader.h"
#include "graphar/filesystem.h"
#include "graphar/graph_info.h"
#include "graphar/reader_util.h"
#include "graphar/types.h"

namespace graphar::ml {
namespace {

constexpr size_t kCurvePoints = 32;

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

IdType Int64ChunkedValueAt(const std::shared_ptr<arrow::ChunkedArray>& col,
                           int64_t idx) {
  int64_t skip = idx;
  for (const auto& chunk : col->chunks()) {
    auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
    if (skip < arr->length()) {
      return static_cast<IdType>(arr->Value(skip));
    }
    skip -= arr->length();
  }
  return 0;
}

Status FillDegreesOrderedByDest(const std::shared_ptr<FileSystem>& fs,
                                const std::string& norm_prefix,
                                const std::shared_ptr<EdgeInfo>& edge_info,
                                std::vector<IdType>* deg) {
  const AdjListType adj = AdjListType::ordered_by_dest;
  const IdType dst_cs = edge_info->GetDstChunkSize();
  GAR_ASSIGN_OR_RAISE(
      IdType vchunks,
      util::GetVertexChunkNum(norm_prefix, edge_info, adj));
  const IdType n = static_cast<IdType>(deg->size());
  for (IdType vc = 0; vc < vchunks; ++vc) {
    GAR_ASSIGN_OR_RAISE(
        auto off_col,
        LoadOffsetColumn(fs, norm_prefix, edge_info, adj, vc));
    const IdType base = vc * dst_cs;
    int64_t m = off_col->length();
    for (IdType local = 0; local < dst_cs && base + local < n; ++local) {
      int64_t i = static_cast<int64_t>(local);
      if (i + 1 >= m) {
        return Status::Invalid("Offset column too short for dst chunk ", vc);
      }
      IdType a = Int64ChunkedValueAt(off_col, i);
      IdType b = Int64ChunkedValueAt(off_col, i + 1);
      (*deg)[base + local] = b - a;
    }
  }
  return Status::OK();
}

Status FillDegreesScanOrderedBySource(const std::string& graph_prefix,
                                      const std::shared_ptr<EdgeInfo>& edge_info,
                                      std::vector<IdType>* deg) {
  const AdjListType adj = AdjListType::ordered_by_source;
  AdjListArrowChunkReader reader(edge_info, adj, graph_prefix);
  std::string norm_prefix;
  GAR_ASSIGN_OR_RAISE(auto fs, FileSystemFromUriOrPath(graph_prefix, &norm_prefix));
  static_cast<void>(fs);
  GAR_ASSIGN_OR_RAISE(IdType src_vchunks,
                      util::GetVertexChunkNum(norm_prefix, edge_info, adj));
  const IdType n = static_cast<IdType>(deg->size());
  for (IdType svc = 0; svc < src_vchunks; ++svc) {
    GAR_ASSIGN_OR_RAISE(
        IdType echunks,
        util::GetEdgeChunkNum(norm_prefix, edge_info, adj, svc));
    for (IdType ec = 0; ec < echunks; ++ec) {
      GAR_RETURN_NOT_OK(reader.seek_chunk_index(svc, ec));
      GAR_ASSIGN_OR_RAISE(auto chunk, reader.GetChunk());
      if (!chunk) continue;
      auto dst_col = chunk->column(1);
      for (int ci = 0; ci < dst_col->num_chunks(); ++ci) {
        auto arr =
            std::static_pointer_cast<arrow::Int64Array>(dst_col->chunk(ci));
        for (int64_t r = 0; r < arr->length(); ++r) {
          IdType dst = arr->Value(r);
          if (dst >= 0 && dst < n) {
            (*deg)[dst]++;
          }
        }
      }
    }
  }
  return Status::OK();
}

std::vector<std::pair<IdType, IdType>> SortedDegIdPairs(
    const std::vector<IdType>& degrees) {
  std::vector<std::pair<IdType, IdType>> out;
  out.reserve(degrees.size());
  for (size_t i = 0; i < degrees.size(); ++i) {
    out.emplace_back(degrees[i], static_cast<IdType>(i));
  }
  std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) {
              if (a.first != b.first) return a.first > b.first;
              return a.second < b.second;
            });
  return out;
}

}  // namespace

DegreeHotNodeSelector::DegreeHotNodeSelector(std::shared_ptr<GraphInfo> graph_info,
                                             std::string vertex_type,
                                             std::string edge_type)
    : graph_info_(std::move(graph_info)),
      vertex_type_(std::move(vertex_type)),
      edge_type_(std::move(edge_type)) {
  edge_info_ = graph_info_->GetEdgeInfo(vertex_type_, edge_type_, vertex_type_);
}

Status DegreeHotNodeSelector::LoadDegrees() {
  if (loaded_) return Status::OK();
  if (!edge_info_) {
    return Status::Invalid("Edge type '", edge_type_,
                           "' not found for vertex type '", vertex_type_, "'");
  }
  auto vertex_info = graph_info_->GetVertexInfo(vertex_type_);
  if (!vertex_info) {
    return Status::Invalid("Vertex type '", vertex_type_, "' not found");
  }

  const std::string& prefix = graph_info_->GetPrefix();
  std::string normalized_prefix;
  GAR_ASSIGN_OR_RAISE(auto fs,
                      FileSystemFromUriOrPath(prefix, &normalized_prefix));
  GAR_ASSIGN_OR_RAISE(num_vertices_,
                      util::GetVertexNum(normalized_prefix, vertex_info));
  degrees_.assign(static_cast<size_t>(num_vertices_), 0);

  if (edge_info_->HasAdjacentListType(AdjListType::ordered_by_dest)) {
    GAR_RETURN_NOT_OK(
        FillDegreesOrderedByDest(fs, normalized_prefix, edge_info_, &degrees_));
  } else if (edge_info_->HasAdjacentListType(AdjListType::ordered_by_source)) {
    GAR_RETURN_NOT_OK(
        FillDegreesScanOrderedBySource(prefix, edge_info_, &degrees_));
  } else {
    return Status::Invalid(
        "Edge type '", edge_type_,
        "' has neither ordered_by_dest nor ordered_by_source; cannot derive "
        "in-degrees");
  }

  loaded_ = true;
  return Status::OK();
}

Result<std::vector<IdType>> DegreeHotNodeSelector::Select(size_t top_k) {
  GAR_RETURN_NOT_OK(LoadDegrees());
  auto sorted = SortedDegIdPairs(degrees_);
  size_t take = std::min(top_k, sorted.size());
  std::vector<IdType> out;
  out.reserve(take);
  for (size_t i = 0; i < take; ++i) {
    out.push_back(sorted[i].second);
  }
  return out;
}

Result<HitRateCurve> DegreeHotNodeSelector::Curve() {
  GAR_RETURN_NOT_OK(LoadDegrees());
  auto sorted = SortedDegIdPairs(degrees_);
  double total = 0;
  for (const auto& d : degrees_) {
    total += static_cast<double>(d);
  }

  HitRateCurve curve;
  if (sorted.empty()) {
    return curve;
  }
  const size_t n = sorted.size();
  if (total <= 0.0) {
    curve.push_back({std::max<size_t>(1, n), 0.0});
    return curve;
  }

  std::vector<double> prefix(n + 1, 0.0);
  for (size_t i = 0; i < n; ++i) {
    prefix[i + 1] = prefix[i] + static_cast<double>(sorted[i].first);
  }
  if (kCurvePoints == 0) {
    return curve;
  }
  size_t last_k = 0;
  for (size_t pi = 0; pi < kCurvePoints; ++pi) {
    double t = kCurvePoints == 1
                   ? 1.0
                   : static_cast<double>(pi) / static_cast<double>(kCurvePoints - 1);
    double lk = (1.0 - t) * std::log(1.0) +
                t * std::log(static_cast<double>(std::max<size_t>(n, 1)));
    size_t k = static_cast<size_t>(std::llround(std::exp(lk)));
    k = std::max<size_t>(1, std::min(k, n));
    if (k == last_k && pi > 0) {
      continue;
    }
    last_k = k;
    curve.push_back({k, prefix[k] / total});
  }
  if (curve.empty() || curve.back().k != n) {
    curve.push_back({n, prefix[n] / total});
  }
  return curve;
}

Result<double> EstimateHitRate(HotNodeSelector& sel, size_t top_k) {
  if (top_k == 0) {
    return 0.0;
  }
  auto* deg_sel = dynamic_cast<DegreeHotNodeSelector*>(&sel);
  if (deg_sel != nullptr) {
    GAR_RETURN_NOT_OK(deg_sel->LoadDegrees());
    const auto& d = deg_sel->degrees_;
    auto sorted = SortedDegIdPairs(d);
    double total = 0;
    for (const auto& x : d) {
      total += static_cast<double>(x);
    }
    if (d.empty()) {
      return 0.0;
    }
    if (total <= 0.0) {
      return 0.0;
    }
    size_t k = std::min(top_k, sorted.size());
    double mass = 0;
    for (size_t i = 0; i < k; ++i) {
      mass += static_cast<double>(sorted[i].first);
    }
    return mass / total;
  }

  GAR_ASSIGN_OR_RAISE(HitRateCurve curve, sel.Curve());
  if (curve.empty()) {
    return Status::Invalid("Empty hit-rate curve");
  }
  if (top_k >= curve.back().k) {
    return curve.back().cumulative_hit_rate;
  }
  size_t i = 0;
  while (i + 1 < curve.size() && curve[i + 1].k <= top_k) {
    ++i;
  }
  if (curve[i].k == top_k) {
    return curve[i].cumulative_hit_rate;
  }
  if (i + 1 >= curve.size()) {
    return curve[i].cumulative_hit_rate;
  }
  const double k0 = static_cast<double>(curve[i].k);
  const double k1 = static_cast<double>(curve[i + 1].k);
  const double y0 = curve[i].cumulative_hit_rate;
  const double y1 = curve[i + 1].cumulative_hit_rate;
  const double tk = static_cast<double>(top_k);
  if (k1 <= k0) {
    return y1;
  }
  return y0 + (y1 - y0) * (tk - k0) / (k1 - k0);
}

}  // namespace graphar::ml
