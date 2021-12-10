#include "covariance_estimator.hxx"
#include "../util/eigen.hxx"
#include "graph.hxx"
#include "visits.hxx"
#include "window.hxx"
#include <iostream>

void covariance_estimator_t::init(const std::vector<size_t> &rungs) {
  // initialize estimates
  _derivatives.init(rungs);
  _ddTs.init(rungs);
  _detcov_weight.resize(rungs.size());
}

void covariance_estimator_t::add_epoch() {
  _derivatives.add_epoch();
  _ddTs.add_epoch();
}

void covariance_estimator_t::drop_epoch() {
  _derivatives.drop_epoch();
  _ddTs.drop_epoch();
}

template <typename value_t>
array_t<value_t> estimate_term(const window_t &free_energies,
                               const history_t<value_t> &estimator,
                               const size_t exclude_epoch) {
  array_t<value_t> estimates(free_energies.num_rungs());
  std::vector<size_t> active_epochs;
  size_t epoch_count = free_energies.num_epochs();
  active_epochs.reserve(epoch_count);
  for (size_t epoch = 0; epoch < epoch_count; epoch++) {
    if (epoch != exclude_epoch && free_energies.count(epoch)) {
      active_epochs.push_back(epoch);
    }
  }
  if (active_epochs.empty()) {
    estimates.fill(value_t());
    return estimates;
  }
  array_t<real_t> epoch_weights(active_epochs.size());

  free_energies.rung_execute([&](size_t ri, size_t /*rung*/) {
    // compute expectations
    for (size_t ei = 0; ei < active_epochs.size(); ei++) {
      epoch_weights[ei] = -free_energies.estimate(active_epochs[ei], ri);
    }
    subtract_maximum_value(epoch_weights);
    real_t total_weight{};
    // correct-sized value
    value_t epoch_weighted_value;
    bool set = false;
    for (size_t ei = 0; ei < active_epochs.size(); ei++) {
      auto epoch = active_epochs[ei];
      real_t weight = free_energies.count(epoch) * std::exp(epoch_weights[ei]);
      if (weight) {
        if (set) {
          epoch_weighted_value += estimator.estimate(epoch, ri) * weight;
        } else {
          epoch_weighted_value = estimator.estimate(epoch, ri) * weight;
        }
      }
      total_weight += weight;
    }
    if (total_weight) {
      epoch_weighted_value /= total_weight;
    }
    estimates[ri] = epoch_weighted_value;
  });
  return estimates;
}

real_t detcov_from_terms(const matrix_t& ddT, const vector_t& d) {
  if (d.size() == 0) { // uninitialized
    return 0;
  }
  matrix_t value = ddT - d * d.transpose();
  return std::sqrt(std::max(real_t(0), determinant(value)));
}

array_t<real_t> covariance_estimator_t::estimate(const window_t &free_energies,
                                                 size_t exclude_epoch) const {
  if (exclude_epoch == size_t(-1)) {
    return expectations();
  }
  array_t<vector_t> d =
      estimate_term(free_energies, _derivatives, exclude_epoch);
  array_t<matrix_t> ddT = estimate_term(free_energies, _ddTs, exclude_epoch);
  array_t<real_t> estimates(d.size());
  free_energies.rung_execute([&](size_t ri, size_t /*rung*/) {
    estimates[ri] = detcov_from_terms(ddT[ri], d[ri]);
  });
  return estimates;
}

template <typename value_t>
void covariance_estimator_t::update_epoch_estimates(
    const graph_t &graph, const window_t &free_energies,
    const std::vector<size_t> &replicas,
    const std::vector<array_t<real_t>> &replica_log_ratios,
    const std::vector<std::vector<value_t>> &values,
    history_t<value_t> &estimator) {
  array_t<real_t> epoch_weights(estimator.num_epochs());
  size_t count = free_energies.count(0);
  size_t replica_count = replicas.size();
  estimator.rung_execute([&](size_t ri, size_t rung) {
    // update current epoch estimate
    auto dim_size = graph.invariance_adjacency[rung].size();
    value_t sum(dim_size, value_t::ColsAtCompileTime == 1 ? 1 : dim_size);
    sum.setZero();
    auto fe = free_energies.estimate(0, ri);
    if (std::isfinite(fe)) {
      for (size_t repi = 0; repi < replica_count; repi++) {
        size_t r = replicas[repi];
        if (!std::isfinite(replica_log_ratios[ri][r])) {
          throw std::out_of_range("Log ratio must be finite");
        }
        auto factor = std::exp(fe + replica_log_ratios[ri][r]);
        sum += factor * values[ri][repi];
      }
    }

    value_t new_estimate = sum;
    if (count > replica_count && std::isfinite(free_energies._deltas[ri])) {
      new_estimate += std::exp(free_energies._deltas[ri]) *
                      estimator.estimate(0, ri) * real_t(count - replica_count);
    }
    if (count) {
      new_estimate /= count;
    }
    if (!new_estimate.allFinite()) {
      throw std::out_of_range("The detcov estimate is not finite");
    }
    estimator.estimate(0, ri) = new_estimate;

    // update expectations
    for (size_t epoch = 0; epoch < estimator.num_epochs(); epoch++) {
      epoch_weights[epoch] = -free_energies.estimate(epoch, ri);
    }
    subtract_maximum_value(epoch_weights);
    real_t total_weight{};
    value_t epoch_weighted_value(
        dim_size, value_t::ColsAtCompileTime == 1 ? 1 : dim_size);
    epoch_weighted_value.setZero();
    for (size_t epoch = 0; epoch < estimator.num_epochs(); epoch++) {
      real_t weight =
          free_energies.count(epoch) * std::exp(epoch_weights[epoch]);
      if (weight) {
        epoch_weighted_value += estimator.estimate(epoch, ri) * weight;
      }
      total_weight += weight;
    }
    if (total_weight) {
      epoch_weighted_value /= total_weight;
    }
    estimator(ri) = epoch_weighted_value;
    if (!estimator(ri).allFinite()) {
      throw std::out_of_range("The detcov expectation is not finite");
    }
  });
}

void covariance_estimator_t::update(
    const graph_t &graph, const window_t &free_energies,
    const std::vector<array_t<real_t>> &replica_log_ratios,
    const std::vector<size_t> &replicas,
    const std::unordered_map<std::string,
                             std::vector<std::unordered_map<size_t, real_t>>>
        &replica_observables) {
  const std::vector<std::unordered_map<size_t, real_t>> &replica_energies =
      replica_observables.at("energy");

  // compute derivatives
  std::vector<std::vector<vector_t>> derivatives(_derivatives.num_rungs());
  _derivatives.rung_execute([&](size_t ri, size_t rung) {
    const auto &rung_adjacency = graph.invariance_adjacency[rung];
    for (auto r : replicas) {
      array_t<real_t> derivative =
          array_t<real_t>::Zero(rung_adjacency.size());
      for (size_t d = 0; d < rung_adjacency.size(); d++) {
        const auto &dim_adjacency = rung_adjacency[d];
        real_t reverse_value = replica_energies[r].at(dim_adjacency[0]);
        real_t forward_value = replica_energies[r].at(dim_adjacency[1]);
        if (dim_adjacency[2] == 0) { // this can hapen with size = 1 window
          derivative[d] = 0;
        } else {
          derivative[d] = (forward_value - reverse_value) / dim_adjacency[2];
        }
      }
      derivatives[ri].push_back(derivative);
    }
  });

  // update derivative epoch estimates
  update_epoch_estimates(graph, free_energies, replicas, replica_log_ratios,
                         derivatives, _derivatives);

  // compute (d - <d>) (d - <d>)^T
  std::vector<std::vector<matrix_t>> ddTs(_ddTs.num_rungs());
  _derivatives.rung_execute([&](size_t ri, size_t /*rung*/) {
    for (size_t repi = 0; repi < replicas.size(); repi++) {
      vector_t d = derivatives[ri][repi];
      ddTs[ri].push_back(d * d.transpose());
    }
  });
  update_epoch_estimates(graph, free_energies, replicas, replica_log_ratios,
                         ddTs, _ddTs);

  // update the weights
  for (size_t ri = 0; ri < _ddTs.num_rungs(); ri++) {
    _detcov_weight[ri] = detcov_from_terms(_ddTs(ri), _derivatives(ri));
  }
}

void covariance_estimator_t::print() {}
