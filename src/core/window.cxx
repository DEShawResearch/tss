#include "window.hxx"
#include "graph.hxx"
#include "../util/eigen.hxx"

void window_t::init(
    const graph_t &graph, const std::vector<size_t> &rungs,
    const std::vector<std::shared_ptr<estimator_t>> &estimator_factories) {
  for (auto estimator_factory : estimator_factories) {
    _estimators[estimator_factory->type()] = estimator_factory->create();
  }
  history_execute([&](auto &e) { e.init(rungs); });
  _log_ratios.resize(num_rungs());
  _deltas.resize(num_rungs());
  _dirty = true;
  _initialized = false;

  // epoch estimation caching
  _rung_start_index.resize(num_rungs() + 1);

  populate_rung_exponents();
  precompute_tilt_sum();

  // cache rung volumes
  _rung_volumes.resize(num_rungs());
  _real_rung_count = 0;
  rung_execute([&](size_t ri, size_t rung) {
    _rung_volumes[ri] = graph.rung_volumes[rung];
    if (_rung_volumes[ri]) {
      _real_rung_count++;
    }
  });
  if (_real_rung_count == 0) {
    throw std::out_of_range("Every window must have a nonzero volume rung");
  }
  _log_rung_volumes = _rung_volumes.log();

  // default the weights to rung volumes
  _weights = _rung_volumes;
  _weights *= 1 / _weights.sum();
  _log_weights = _weights.log();
}

void window_t::add_epoch() {
  history_execute([&](auto &e) { e.add_epoch(); });
  populate_rung_exponents();
  precompute_tilt_sum();
}

void window_t::drop_epoch() {
  _dirty = true;
  history_execute([&](auto &e) { e.drop_epoch(); });
  populate_rung_exponents();
  precompute_tilt_sum();
}

void window_t::clear_ratios(size_t replica_count) {
  for (auto &lr : _log_ratios) {
    lr.setConstant(replica_count, -inf<real_t>());
  }
  _active_replicas.clear();
}

array_t<real_t>
window_t::compute_actions(const std::unordered_map<size_t, real_t> &energies) {
  array_t<real_t> actions(num_rungs());
  tss_log_printf("DENOM\n");
  if (_initialized) {
    rung_execute([&](size_t ri, size_t rung) {
      if (std::isfinite(_log_weights[ri])) {
        actions[ri] = (*this)(ri)-energies.at(rung) + _log_weights[ri];
      } else {
        actions[ri] = -inf<real_t>();
      }
      tss_log_printf("%f (%f, %f, %f) ", actions[ri], (*this)(ri),
                     energies.at(rung), _log_weights[ri]);
    });
  } else {
    actions.fill(real_t() - std::log(num_rungs()));
  }
  return actions;
}

void window_t::compute_importance_ratios(
    const std::unordered_map<size_t, real_t> &energies,
    const size_t replica_index, const real_t log_mixture_distribution) {
  _active_replicas.push_back(replica_index);
  tss_log_printf("\nACTIONS\n");
  rung_execute([&](size_t ri, size_t rung) {
    _log_ratios[ri][replica_index] =
        -energies.at(rung) - log_mixture_distribution;
    tss_log_printf(
        "%lu: %f (%f, %f, %f, %f)\n", rung, _log_ratios[ri][replica_index],
        estimate(0, ri) + _log_ratios[ri][replica_index], estimate(0, ri),
        energies.at(rung), log_mixture_distribution);
    check_finite(_log_ratios[ri][replica_index]);
  });
  tss_log_printf("\n");
}

void window_t::update_epoch_fe_estimates() {
  const real_t upper_limit = -std::log(std::numeric_limits<real_t>::epsilon());
  auto replica_count = _active_replicas.size();
  if (replica_count == 0) {
    _deltas.fill(real_t());
    return;
  }
  count(0) += replica_count;
  size_t n = count(0);
  _dirty = true;
  real_t epoch_factor = 1. / n;
  real_t log_epoch_factor = std::log(epoch_factor);

  // per-rung updates
  rung_execute([&](size_t ri, size_t /*rung*/) {
    real_t log_ratio = log_sum_exp(_log_ratios[ri]);
    real_t& rung_estimate = estimate(0, ri);
    real_t& delta = _deltas[ri];

    // update free energy estimates
    if (!std::isfinite(rung_estimate)) {
      rung_estimate = -(log_epoch_factor + log_ratio);
      delta = -inf<real_t>();
    } else {
      if (n > replica_count &&
          rung_estimate + log_epoch_factor + log_ratio < upper_limit) {
        delta =
            -std::log1p(epoch_factor *
                        (std::exp(rung_estimate + log_ratio) - replica_count));
      } else {
        delta = -(rung_estimate + log_epoch_factor + log_ratio);
      }
      rung_estimate += delta;
    }
    check_finite(rung_estimate, "estimate is not finite");
  });
}

void window_t::get_overall_fe_estimates() {
  if (_dirty) {
    _expectations = estimate_all_epochs();
    // check whether all are finite
    _initialized = _expectations.isFinite().all();
  }
}

// cache all epoch count information aside from the current epoch
void window_t::populate_rung_exponents() {
  Eigen::Index static_epochs = num_epochs() - 1;
  array_t<real_t> rung_partial_sums =
      array_t<real_t>::Constant(num_rungs(), -inf<real_t>());
  _total_count = 0;
  // if we have any epochs that are full, cache their contributions
  if (static_epochs > 0) {
    array_t<real_t> epoch_counts(static_epochs);
    for (Eigen::Index epoch = 0; epoch < static_epochs; epoch++) {
      epoch_counts[epoch] = count(epoch + 1);
      _total_count += epoch_counts[epoch];
    }
    array_t<real_t> log_epoch_counts = epoch_counts.log();

    array_t<real_t> epoch_rung_exponents =
        array_t<real_t>::Constant(static_epochs * num_rungs(), -inf<real_t>());
    std::vector<size_t> rung_start_index;
    fill_with_strided_indices(rung_start_index, num_rungs(), static_epochs);
    for (Eigen::Index epoch = 0; epoch < static_epochs; epoch++) {
      if (epoch_counts[epoch]) {
        for (size_t ri = 0; ri < num_rungs(); ri++) {
          epoch_rung_exponents[ri * static_epochs + epoch] =
            -estimate(epoch + 1, ri) + log_epoch_counts[epoch];
        }
      }
    }
    log_sum_exp(rung_start_index, epoch_rung_exponents, rung_partial_sums);
  }

  // pre-populate the per-rung sums for use in estimate_all_epochs()
  _epoch_rung_exponents.resize(num_rungs() * 2);
  fill_with_strided_indices(_rung_start_index, num_rungs(), 2);
  for (size_t ri = 0; ri < num_rungs(); ri++) {
    _epoch_rung_exponents[2 * ri + 1] = rung_partial_sums[ri];
  }
}

void window_t::precompute_tilt_sum() {
  _cached_total_tilt.setConstant(num_rungs(), 0);
  for (size_t epoch = 1; epoch < num_epochs(); epoch++) {
    _cached_total_tilt += _tilts.count(epoch) * _tilts.estimates(epoch);
  }
}

array_t<real_t> window_t::estimate_all_epochs() {
  size_t current_epoch_count = count(0);
  if (_total_count + current_epoch_count == 0) {
    return array_t<real_t>::Constant(num_rungs(), inf<real_t>());
  }
  if (current_epoch_count) {
    real_t log_count = std::log(current_epoch_count);
    for (size_t ri = 0; ri < num_rungs(); ri++) {
      _epoch_rung_exponents[_rung_start_index[ri]] =
          -estimate(0, ri) + log_count;
    }
  }
  log_sum_exp(_rung_start_index, _epoch_rung_exponents, aggregate_estimate);
  real_t log_total_count = std::log(_total_count + current_epoch_count);
  return log_total_count - aggregate_estimate;
}

array_t<real_t> window_t::expectations(const size_t exclude_epoch) const {
  if (exclude_epoch == size_t(-1)) {
    return _expectations;
  }
  array_t<real_t> epoch_counts(num_epochs());
  real_t total_count = 0;
  std::vector<size_t> active_epochs;
  active_epochs.reserve(num_epochs());
  for (size_t epoch = 0; epoch < num_epochs(); epoch++) {
    epoch_counts[epoch] = count(epoch);
    if (epoch != exclude_epoch) {
      total_count += epoch_counts[epoch];
      if (epoch_counts[epoch]) {
        active_epochs.push_back(epoch);
      }
    }
  }
  if (total_count == 0) {
    return array_t<real_t>::Constant(num_rungs(), inf<real_t>());
  }

  // compute log of counts, vectorized
  array_t<real_t> log_epoch_counts = epoch_counts.log();
  real_t log_total_count = std::log(total_count);

  rung_exponents.resize(active_epochs.size() * num_rungs());
  fill_with_strided_indices(rung_start_index, num_rungs(),
                            active_epochs.size());

  rung_execute([&](size_t ri, size_t /*rung*/) {
    for (size_t ei = 0; ei < active_epochs.size(); ei++) {
      auto epoch = active_epochs[ei];
      auto log_count = log_epoch_counts[epoch];
      real_t rung_estimate = estimate(epoch, ri);
      rung_exponents[ri * active_epochs.size() + ei] =
          -rung_estimate + log_count;
    }
  });
  log_sum_exp(rung_start_index, rung_exponents, aggregate_estimate);
  return log_total_count - aggregate_estimate;
}

void window_t::update_tilts(const std::vector<size_t>& rung_visits) {
  array_t<real_t> visit_contribution = array_t<real_t>::Zero(num_rungs());
  for (auto ri : rung_visits) {
    visit_contribution[ri] += 1 / _weights[ri];
    tss_log_printf("Rung %lu inv weight: %f tilt: %f\n", ri,
                   1 / _weights[ri], _tilts.estimates(0)[ri]);
  }
  real_t n_prev = _tilts.count(0);
  _tilts.count(0) += rung_visits.size();
  real_t n_new = _tilts.count(0);
  _tilts.estimates(0) +=
      1 / n_new * ((n_prev - n_new) * _tilts.estimates(0) + visit_contribution);
}

void window_t::update_non_fe_estimates(
    const graph_t &graph,
    const std::unordered_map<std::string,
                             std::vector<std::unordered_map<size_t, real_t>>>
        &replica_observables) {
  if (_dirty) {
    for (auto& estimator : _estimators) {
      estimator.second->update(graph, *this, _log_ratios, _active_replicas,
                               replica_observables);
    }
    _dirty = false;
  }
}

array_t<real_t>
window_t::estimated_detcov(size_t exclude_epoch) const {
  return logged_at(_estimators, estimator_type_t::detcov(), "_estimators")
      ->estimate(*this, exclude_epoch);
}

real_t window_t::max_estimated_detcov(size_t exclude_epoch) const {
  if (!_initialized) {
    return 0;
  }
  return estimated_detcov(exclude_epoch).maxCoeff();
}

array_t<real_t> window_t::overall_tilt(size_t exclude_epoch) const {
  array_t<real_t> total_tilt;
  real_t total_count = 0;
  if (exclude_epoch == size_t(-1)) {
    size_t c = _tilts.count(0);
    total_tilt = _cached_total_tilt + c * _tilts.estimates(0);
    total_count = _total_count + c;
  } else {
    total_tilt = array_t<real_t>::Zero(num_rungs());
    for (size_t epoch = 0; epoch < num_epochs(); epoch++) {
      if (epoch == exclude_epoch) {
        continue;
      }
      size_t c = _tilts.count(epoch);
      total_tilt += c * _tilts.estimates(epoch);
      total_count += c;
    }
  }
  if (total_count) {
    return total_tilt / total_count;
  }
  return array_t<real_t>::Ones(num_rungs());
}

array_t<real_t> window_t::normalized_weights(real_t detcov_epsilon,
                                             real_t max_detcov,
                                             size_t exclude_epoch) {
  array_t<real_t> detcov = estimated_detcov(exclude_epoch);

  // Apply the weight regularization
  array_t<real_t> weights =
      ((1 - detcov_epsilon) * detcov + detcov_epsilon * max_detcov) *
      _rung_volumes;
  real_t total = weights.sum();
  if (total) {
    weights *= 1 / total;
  } else {
    real_t one_over_rrc = 1.0 / _real_rung_count;
    rung_execute([&](size_t ri, size_t /*rung*/) {
      weights[ri] = _rung_volumes[ri] ? one_over_rrc : 0;
    });
  }
  return weights;
}
