#pragma once

#include "estimator.hxx"
#include "history.hxx"
#include "../util/eigen.hxx"

class graph_t;

class window_t : public history_t<real_t> {
  std::vector<size_t> _active_replicas;
  bool _dirty;
  size_t _total_count;
  std::vector<size_t> _rung_start_index;
  array_t<real_t> _epoch_rung_exponents;
  std::vector<array_t<real_t>> _log_ratios;
  array_t<real_t> _cached_total_tilt;

  // temporaries
  mutable array_t<real_t> rung_exponents;
  mutable std::vector<size_t> rung_start_index;
  mutable array_t<real_t> maxes;
  mutable array_t<real_t> aggregate_estimate;

  template <typename epoch_function_t>
  void history_execute(epoch_function_t &&f) {
    f(*(history_t<real_t>*)(this));
    f(_tilts);
    for (auto &estimator : _estimators) {
      f(*estimator.second);
    }
  }

public:
  history_t<real_t> _tilts;
  std::unordered_map<std::string, std::shared_ptr<estimator_t>> _estimators;
  array_t<real_t> _log_weights;
  array_t<real_t> _weights;
  array_t<real_t> _log_normalized_weights;
  array_t<real_t> _deltas;
  array_t<real_t> _rung_volumes;
  array_t<real_t> _log_rung_volumes;
  size_t _real_rung_count;
  bool _initialized;

  window_t() : history_t<real_t>(inf<real_t>()) {}

  void
  init(const graph_t &graph, const std::vector<size_t> &rungs,
       const std::vector<std::shared_ptr<estimator_t>> &estimator_factories);

  bool initialized() const { return _initialized; }

  // epoch handling
  void add_epoch();
  void drop_epoch();
  void populate_rung_exponents();
  void precompute_tilt_sum();

  // free energy update
  // 1. prepare the window to start an update
  void clear_ratios(size_t replica_count);
  // 2. compute the actions (used to compute mixture denominator)
  array_t<real_t>
  compute_actions(const std::unordered_map<size_t, real_t> &energies);
  // 3. compute per-replica importance ratios given the mixture denominator
  void
  compute_importance_ratios(const std::unordered_map<size_t, real_t> &energies,
                            const size_t replica_index,
                            const real_t log_mixture_distribution);
  // 4. use importance ratios from all replicas to update rung FE estimates
  void update_epoch_fe_estimates();
  // 5. get overall rung FE estimates by combining estimates from all epochs
  void get_overall_fe_estimates();

  // free energy helper methods to combine estimates from all or some epochs
  array_t<real_t> estimate_all_epochs();
  array_t<real_t> expectations(const size_t exclude_epoch) const;

  // tilt updates
  void update_tilts(const std::vector<size_t>& rung_visits);
  
  // weight-related methods
  void update_non_fe_estimates(
      const graph_t &graph,
      const std::unordered_map<std::string,
                               std::vector<std::unordered_map<size_t, real_t>>>
          &replica_observables);
  array_t<real_t> estimated_detcov(size_t exclude_epoch = -1) const;
  real_t max_estimated_detcov(size_t exclude_epoch) const;
  array_t<real_t> overall_tilt(size_t exclude_epoch = -1) const;
  array_t<real_t> normalized_weights(real_t detcov_epsilon, real_t max_detcov,
                                     size_t exclude_epoch = -1);
};
