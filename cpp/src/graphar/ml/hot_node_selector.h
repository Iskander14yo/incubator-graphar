#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "graphar/fwd.h"

namespace graphar::ml {

struct HitRateCurvePoint {
  size_t k;
  double cumulative_hit_rate;
};

using HitRateCurve = std::vector<HitRateCurvePoint>;

class HotNodeSelector {
 public:
  virtual ~HotNodeSelector() = default;
  virtual Result<std::vector<IdType>> Select(size_t top_k) = 0;
  virtual Result<HitRateCurve> Curve() = 0;
};

class DegreeHotNodeSelector : public HotNodeSelector {
 public:
  DegreeHotNodeSelector(std::shared_ptr<GraphInfo> graph_info,
                        std::string vertex_type, std::string edge_type);

  Result<std::vector<IdType>> Select(size_t top_k) override;
  Result<HitRateCurve> Curve() override;

 private:
  friend Result<double> EstimateHitRate(HotNodeSelector& sel, size_t top_k);

  Status LoadDegrees();

  std::shared_ptr<GraphInfo> graph_info_;
  std::string vertex_type_;
  std::string edge_type_;
  std::shared_ptr<EdgeInfo> edge_info_;
  std::vector<IdType> degrees_;
  IdType num_vertices_{0};
  bool loaded_{false};
};

Result<double> EstimateHitRate(HotNodeSelector& sel, size_t top_k);

}  // namespace graphar::ml
