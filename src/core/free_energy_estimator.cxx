#include "../util/eigen.hxx"
#include <cassert>
#include <iostream>
#include <queue>
#include "free_energy_estimator.hxx"
#include "epochs.hxx"
#include "graph.hxx"
#include "visits.hxx"

void free_energy_estimator_t::init(
    const graph_t &graph,
    const std::vector<std::shared_ptr<estimator_t>> &estimator_factories) {
  size_t window_count = graph.window_rungs.size();
  _windows.resize(window_count);
  for (size_t wi = 0; wi < window_count; wi++) {
    _windows[wi].init(graph, graph.window_rungs[wi], estimator_factories);
  }
  _expectations.setConstant(graph.rung_count(), inf<real_t>());
  _weights.setConstant(graph.rung_count(), real_t());
  _weight_epsilon = 1e-2;
  _log_epsilon_pi = std::log(1e-3);
  _log_one_minus_epsilon_pi = std::log(1 - 1e-3);
  _rung_count = graph.rung_count();

  // make sure we're at least estimating the end-to-end error
  _pair_errors[{0, _rung_count - 1}] = inf<real_t>();

  _rung_errors.setConstant(_rung_count, real_t());
  _f.setConstant(_windows.size(), real_t());

  _visit_control_eta = 2;

  // nonlinear solve parameters and reporting
  _solver_iterations = 60;
  _global_solver_tolerance = 1e-5;
  _last_solve_iterations = 0;
  _last_solve_error = inf<real_t>();
  _omit_solve = false;
}

void free_energy_estimator_t::add_epoch() {
  for (auto& window : _windows) {
    window.add_epoch();
  }
}

void free_energy_estimator_t::drop_epoch() {
  for (auto& window : _windows) {
    window.drop_epoch();
  }
}

void free_energy_estimator_t::update_estimates(
    const graph_t &graph,
    const std::unordered_map<std::string,
                             std::vector<std::unordered_map<size_t, real_t>>>
        &replica_observables,
    const std::vector<size_t> &replica_windows) {
  get_importance_ratios(replica_observables.at("energy"), replica_windows);
  window_execute([&](auto &window, size_t /*wi*/) {
    window.update_epoch_fe_estimates();
    window.get_overall_fe_estimates();
    window.update_non_fe_estimates(graph, replica_observables);
  });

  // the _omit_solve flag is purely for testing purposes
  if (!_omit_solve) {
    // update window weights
    if (_visit_control_eta) {
      // get globally consistent FEs and use them to update weights
      update_global_estimates(graph);
    } else {
      get_window_data(data, graph, false);
      window_execute([&](auto &window, size_t wi) {
        window._weights = data.window_weights[wi];
        window._log_weights = data.log_window_weights[wi];
      });
    }
  }
}

void free_energy_estimator_t::update_tilts(
    const std::unordered_map<size_t, std::vector<size_t>> &window_rung_visits) {
  for (const auto& rung_visits : window_rung_visits) {
    _windows[rung_visits.first].update_tilts(rung_visits.second);
  }
}

void free_energy_estimator_t::get_window_values(windows_data_t &data,
                                                real_t eta,
                                                const size_t exclude_epoch) {
  size_t window_count = _windows.size();
  data.log_window_weights.resize(window_count);
  data.window_weights.resize(window_count);
  data.window_fes.resize(window_count);
  data.tilts.resize(window_count);

  // compute max detcov
  real_t max_detcov = 0;
  for (auto &window : _windows) {
    max_detcov =
        std::max(max_detcov, window.max_estimated_detcov(exclude_epoch));
  }

  window_execute([&](auto &window, size_t wi) {
    data.window_fes[wi] = window.expectations(exclude_epoch);
    data.window_weights[wi] =
        window.normalized_weights(_weight_epsilon, max_detcov, exclude_epoch);
    data.log_window_weights[wi] = data.window_weights[wi].log();
    
    if (std::isfinite(eta)) {
      data.tilts[wi] = window.overall_tilt(exclude_epoch);
    } else {
      data.tilts[wi].setConstant(window.num_rungs(), 1.0);
    }
  });
  data.eta_plus_one = eta + 1;
}

/*
  Update the jackknife mean squared error (MSE) computation.  Compute estimates
  for each rung FE - the mean FE both using all epochs and with each epoch
  dropped out and use the results to estimate the MSE.  Do the same for FE
  differences between pairs of rungs.  Note that this is only well-defined
  when all epochs have many samples; ideally use this only when all epochs
  are full.
 */
void free_energy_estimator_t::update_errors(const graph_t &graph,
                                            const epoch_t &epochs) {
  size_t num_epochs = epochs.count();
  assert(_windows[0].num_epochs() == num_epochs);
  // get epoch weights
  auto epoch_weights = epochs.weights();

  _expectations = solve_fes_infinite_eta(data, graph);
  // get pairwise delta FEs using all epochs
  pair_to_value_map_t pairwise_delta_fe;
  for (const auto& rung_pair_error : _pair_errors) {
    const auto& rung_pair = rung_pair_error.first;
    pairwise_delta_fe[rung_pair] =
        _expectations[rung_pair.second] - _expectations[rung_pair.first];
    // initialize to zero
    _pair_errors[rung_pair] = real_t();
  }
  // initialize computation of overall errors
  _rung_errors.setConstant(_rung_count, real_t());
  array_t<real_t> overall_fes_minus_mean = _expectations - _expectations.mean();

  // compute epoch error contributions
  array_t<real_t> epoch_fes;
  for (size_t epoch = 0; epoch < num_epochs; epoch++) {
    epoch_fes = solve_fes_infinite_eta(data, graph, epoch);

    auto w = epoch_weights[epoch];
    real_t factor = std::pow(1 - w, 2) / w;

    for (const auto& rung_pair_error : _pair_errors) {
      const auto& rung_pair = rung_pair_error.first;
      real_t delta_fe =
          epoch_fes[rung_pair.second] - epoch_fes[rung_pair.first];
      _pair_errors[rung_pair] +=
          factor * std::pow(pairwise_delta_fe[rung_pair] - delta_fe, 2);
    }

    _rung_errors += factor * Eigen::square((epoch_fes - epoch_fes.mean()) -
                                      overall_fes_minus_mean);
  }

  // scale and convert from MSE to RMSE
  real_t epoch_scale = real_t(1) / (num_epochs - 1);
  _rung_errors = (_rung_errors * epoch_scale).sqrt();
  for (auto &pe : _pair_errors) {
    pe.second = std::sqrt(pe.second * epoch_scale);
  }
}

void free_energy_estimator_t::get_active_windows(windows_data_t &data,
                                                 const graph_t &graph) const {
  size_t window_count = _windows.size();
  data.rung_window_count.resize(_rung_count);
  data.rung_window_count.fill(0);

  // 1. See which windows _might_ be active (must have some rung with nonzero
  //    and must not have any infinite FEs).
  //    Count how many active windows have a nonzero weight at each rung.
  std::vector<bool> active_window_candidate(window_count, false);
  window_execute([&](const auto &window, size_t wi) {
    // check whether this window could be active
    const auto &lww = data.log_window_weights[wi];
    bool has_infinite_fe = !data.window_fes[wi].isFinite().all();
    bool has_finite_log_weight =
        (data.window_weights[wi] * data.tilts[wi]).sum() > 0;
    // if it could, contribute to active rungs
    if (window.initialized() && !has_infinite_fe && has_finite_log_weight) {
      active_window_candidate[wi] = true;
      window.rung_execute([&](size_t ri, size_t rung) {
        if (lww[ri] != -inf<real_t>()) {
          data.rung_window_count[rung]++;
        }
      });
    }
  });

  // 2. If require_all_windows is true, we only record rungs which are active
  //    in all overlapping windows.  Otherwise, we record rungs that are active
  //    in _any_ overlapping window.  If these measures don't match (which
  //    should only be true before the graph has been fully explored), we may
  //    later need to recompute FEs after solving for window shifts.
  data.active_rungs.clear();
  data.active_rungs.reserve(_rung_count);
  data.active_rung_mask.resize(_rung_count);
  fill(data.active_rung_mask, false);

  std::vector<bool> active_window_mask(window_count, false);
  for (size_t rung = 0; rung < _rung_count; rung++) {
    if (data.rung_window_count[rung] > 0) {
      for (auto wi : graph.rung_to_windows[rung]) {
        active_window_mask[wi] = true;
      }
      data.active_rung_mask[rung] = true;
      data.active_rungs.push_back(rung);
    }
  }

  // 3. Record the set of windows that we are considering to be active (windows
  //    that contain active rungs according to the criteria in 2) and create
  //    appropriate index mappings.
  data.window_map.resize(window_count);
  fill(data.window_map, -1);
  data.active_windows.clear();
  data.active_windows.reserve(window_count);
  data.window_rung_count = 0;

  for (size_t wi = 0; wi < window_count; wi++) {
    if (active_window_mask[wi] && active_window_candidate[wi]) {
      data.window_map[wi] = data.active_windows.size();
      data.active_windows.push_back(wi);
      data.window_rung_count += _windows[wi].num_rungs();
    }
  }  
}

void free_energy_estimator_t::get_window_data(windows_data_t &data,
                                              const graph_t &graph,
                                              real_t eta,
                                              const size_t exclude_epoch) {
  // get per-window weight and FE values
  get_window_values(data, eta, exclude_epoch);
  get_active_windows(data, graph);
  if (eta > 0) {
    // compute empirical window weights
    window_weights_qr(data, graph);
    // compute empirical rung weights
    global_weights(data);
  }
}

void free_energy_estimator_t::window_weights_qr(windows_data_t &data,
                                                const graph_t &graph) {
  data.pi.resize(_windows.size());
  size_t active_window_count = data.active_windows.size();
  if (active_window_count == 0) {
    data.pi.fill(0);
    return;
  }
  if (active_window_count == 1) {
    data.pi.fill(0);
    data.pi[data.active_windows[0]] = 1;
    return;
  }
  // set up system of equations for the weights
  triplet_list_t triplet_list;
  triplet_list.reserve(active_window_count * 2);
  for (auto wi : data.active_windows) {
    int row = data.window_map[wi];
    real_t inv_norm = 1 / (data.window_weights[wi] * data.tilts[wi]).sum();
    // if there are any rungs where the other window isn't active, that weight
    // goes to this window instead (and thus appears on the diagonal).  That's
    // because the probability of the rung appearing in this window is 1 rather
    // than 1/2 in that case.
    real_t diagonal_term = 0;

    // compute contribution due to each rung in window
    _windows[wi].rung_execute([&](size_t ri, size_t rung) {
      real_t weight =
          data.window_weights[wi][ri] * data.tilts[wi][ri] * inv_norm;
      auto other_row = data.window_map[graph.swap_window(wi, rung)];
      if (other_row == -1) {
        diagonal_term += weight;
      }
      else if (other_row > 0) {
        // do not fill in first row, because we will replace it with ones
        triplet_list.push_back({other_row, row, weight});
      }
    });
    // do not fill in first row
    if (row > 0) {
      triplet_list.push_back({row, row, diagonal_term - 1});
    }
    // top row all ones
    triplet_list.push_back({0, row, 1});
  }

  // build P
  sparse_matrix_t P = build_sparse_matrix(triplet_list, active_window_count);
  // Solve P pi = (1, 0, ..., 0)
  vector_t rhs = vector_t::Zero(active_window_count);
  rhs[0] = 1;
  vector_t pi = solve_qr(P, rhs);

  data.pi.fill(0);
  for (auto wi : data.active_windows) {
    // clamp wpi to be positive - NaNs will appear downstream if this isn't done
    data.pi[wi] = std::max(std::numeric_limits<real_t>::epsilon(),
                           pi(data.window_map[wi]));
  }
  data.pi *= 1 / data.pi.sum();
}

array_t<real_t> free_energy_estimator_t::solve_fes_infinite_eta(
    windows_data_t &data, const graph_t &graph, size_t exclude_epoch) {
  get_window_data(data, graph, inf<real_t>(), exclude_epoch);
  
  size_t active_window_count = data.active_windows.size();
  if (active_window_count == 0) {
    return array_t<real_t>::Constant(_rung_count, inf<real_t>());
  }

  auto active = [&](size_t wi) { return data.window_map[wi] != -1; };
  auto rung_active = [&](size_t rung) { return data.active_rung_mask[rung]; };

  triplet_list_t triplet_list;
  triplet_list.reserve(active_window_count * 2);
  for (const auto& intersection : graph.intersecting_windows) {
        size_t wi, wj;
    std::tie(wi, wj) = intersection.first;
    if (!active(wi) || !active(wj)) {
      continue;
    }
    int awi = data.window_map[wi];
    int awj = data.window_map[wj];
    const auto &wwi = data.window_weights[wi];
    const auto &wwj = data.window_weights[wj];

    const auto &intersection_set = intersection.second;
    real_t tij = 0;
    for (size_t ri = 0; ri < intersection_set.size(); ri++) {
      size_t rung, rwi, rwj;
      std::tie(rung, rwi, rwj) = intersection_set[ri];
      if (rung_active(rung)) {
        tij += wwi[rwi] * wwj[rwj] / data.p[rung];
      }
    }
    triplet_list.push_back({awj, awi, -tij * data.pi[wi]});
    triplet_list.push_back({awi, awj, -tij * data.pi[wj]});
  }

  array_t<real_t> fe_rung;
  fe_rung.setConstant(_rung_count, real_t());
  window_list_execute(data.active_windows, [&](auto &window, size_t wi,
                                               size_t /*awi*/) {
    const auto &wwi = data.window_weights[wi];
    const auto &wfei = data.window_fes[wi];
    window.rung_execute([&](size_t ri, size_t rung) {
      if (rung_active(rung)) {
        fe_rung[rung] += data.pi[wi] * wwi[ri] * wfei[ri] / data.p[rung];
      }
    });
  });  

  vector_t rhs = vector_t::Zero(active_window_count + 1);
  window_list_execute(data.active_windows, [&](auto &window, size_t wi,
                                               int awi) {
    const auto &wwi = data.window_weights[wi];
    const auto &wfei = data.window_fes[wi];
    triplet_list.push_back({(int)active_window_count, awi, data.pi[wi]});
    real_t tij = 0;
    window.rung_execute([&](size_t ri, size_t rung) {
      if (rung_active(rung)) {
        rhs[awi] += wwi[ri] * (wfei[ri] - fe_rung[rung]);
        tij += data.pi[wi] * wwi[ri] * wwi[ri] / data.p[rung];
      }
    });
    triplet_list.push_back({awi, awi, 1 - tij});
  });

  // build T
  sparse_matrix_t I_minus_T = build_sparse_matrix(
      triplet_list, active_window_count + 1, active_window_count);
  // Solve (I - T) f = ...
  vector_t f = solve_qr(I_minus_T, rhs);

  array_t<real_t> fe_rung_reported;
  fe_rung_reported.setConstant(_rung_count, inf<real_t>());
  for (auto rung : data.active_rungs) {
    fe_rung_reported[rung] = 0;
  }
  window_list_execute(data.active_windows, [&](auto &window, size_t wi,
                                               size_t awi) {
    const auto &wwi = data.window_weights[wi];
    const auto &wfei = data.window_fes[wi];
    window.rung_execute([&](size_t ri, size_t rung) {
      if (rung_active(rung)) {
        fe_rung_reported[rung] +=
            data.pi[wi] * wwi[ri] * (wfei[ri] - f[awi]) / data.p[rung];
      }
    });
  });

  return fe_rung_reported;
}

void free_energy_estimator_t::global_weights(windows_data_t &data) {
  // set up defaults
  data.p.resize(_rung_count);
  data.p.fill(quiet_nan<real_t>());

  if (data.active_windows.empty()) {
    return;
  }

  // for each active window, accumulate rung weight contributions
  // initialize values and filter windows
  for (auto awi : data.active_windows) {
    for (auto rung : _windows[awi]._rungs) {
      data.p[rung] = 0;
    }
  }

  for (auto wi : data.active_windows) {
    real_t wpi = data.pi[wi];
    real_t inv_norm = 1 / (data.window_weights[wi] * data.tilts[wi]).sum();
    _windows[wi].rung_execute([&](size_t ri, size_t rung) {
      data.p[rung] +=
          wpi * data.window_weights[wi][ri] * data.tilts[wi][ri] * inv_norm;
    });
  }

  // Note that this recalculation is unnecessary if Qp=p is solved to sufficient
  // precision
  for (auto wi : data.active_windows) {
    data.pi[wi] = 0;
    _windows[wi].rung_execute([&](size_t /*ri*/, size_t rung) {
      // only include active rungs
      if (data.rung_window_count[rung]) {
        data.pi[wi] += data.p[rung] / data.rung_window_count[rung];
      }
    });
  }
  // explicitly normalize
  data.pi *= 1 / data.pi.sum();
  // precompute logs
  data.log_p = data.p.log();
  data.log_pi = data.pi.log();
}

// compute the gradient and one-norm of the window shift optimization objective
real_t compute_onenorm(const windows_data_t &data, const array_t<real_t> &f,
                       const array_t<real_t> &hf) {
  size_t active_window_count = data.active_windows.size();
  auto &window_grad_part = eigen_memory_pool_t::get_real(active_window_count);
  for (size_t awi = 0; awi < active_window_count; awi++) {
    auto wi = data.active_windows[awi];
    window_grad_part[awi] = (hf[wi] - f[wi]) / data.eta_plus_one;
  }
  window_grad_part = window_grad_part.exp();
  vector_t grad(active_window_count);
  for (size_t awi = 0; awi < active_window_count; awi++) {
    grad[awi] = data.pi[data.active_windows[awi]] *
                (1 - window_grad_part[awi]) / data.eta_plus_one;
  }
  return grad.cwiseAbs().sum();
}

void free_energy_estimator_t::compute_rung_fes_from_shifts(
    const array_t<real_t> &f, const windows_data_t &data,
    array_t<real_t> &computed_rung_fes) {
  // build _working_fes from shifts
  auto &rung_exponents = eigen_memory_pool_t::get_real(2 * _rung_count);
  // This is a hack to achieve the following: if we have no information for a
  // rung, its estimate is +inf.  If we have one estimate, the estimate is that.
  // If we have two estimates, they get added together.
  for (size_t rung = 0; rung < _rung_count; rung++) {
    rung_exponents[2 * rung] = inf<real_t>();
    rung_exponents[2 * rung + 1] = -inf<real_t>();
  }
  
  rung_visited.resize(_rung_count);
  fill(rung_visited, 0);
  fill_with_strided_indices(start, _rung_count, 2);
  for (auto wi : data.active_windows) {
    const auto &wfe = data.window_fes[wi];
    // collect contributions to each rung
    _windows[wi].rung_execute([&](size_t ri, size_t rung) {
      real_t log_rung_weight = data.log_window_weights[wi][ri];
      if (std::isfinite(log_rung_weight) && std::isfinite(data.log_p[rung])) {
        rung_exponents[2 * rung + rung_visited[rung]++] =
            data.log_pi[wi] + (wfe[ri] - f[wi]) / data.eta_plus_one +
            log_rung_weight - data.log_p[rung];
      }
    });
  }

  // accumulate value for each rung
  log_sum_exp(start, rung_exponents, computed_rung_fes);
  computed_rung_fes *= data.eta_plus_one;
}

array_t<real_t>
free_energy_estimator_t::iterate_f(const array_t<real_t> &rung_fes,
                                   const windows_data_t &data) {
  auto &window_exponents =
      eigen_memory_pool_t::get_real(data.window_rung_count);
  window_exponents.fill(-inf<real_t>());
  fill_with_window_rung_counts(start, data.active_windows);
  window_list_execute(data.active_windows, [&](auto &window, size_t wi,
                                               size_t awi) {
    const auto &lww = data.log_window_weights[wi];
    const auto &wfe = data.window_fes[wi];
    window.rung_execute([&](size_t ri, size_t rung) {
      if (data.active_rung_mask[rung]) {
        window_exponents[start[awi] + ri] =
            lww[ri] + (wfe[ri] - rung_fes[rung]) / data.eta_plus_one;
      }
    });
  });

  size_t active_window_count = data.active_windows.size();
  array_t<real_t> window_fs(active_window_count);
  log_sum_exp(start, window_exponents, window_fs);

  array_t<real_t> final_fs =
      array_t<real_t>::Constant(_windows.size(), real_t());
  for (size_t awi = 0; awi < active_window_count; awi++) {
    final_fs[data.active_windows[awi]] = window_fs[awi];
  }

  return data.eta_plus_one * final_fs;
}

array_t<real_t> normalized_f(const array_t<real_t> &f,
                             const windows_data_t &data) {
  real_t weighted_sum = 0;
  for (auto wi : data.active_windows) {
    weighted_sum += data.pi[wi] * f[wi];
  }
  return f - weighted_sum;
}

std::tuple<array_t<real_t>, array_t<real_t>, size_t, real_t>
free_energy_estimator_t::improve_fes(const graph_t &graph,
                                     const array_t<real_t> &f_input,
                                     size_t global_iterations) {
  // get per-window values
  get_window_data(data, graph, _visit_control_eta);

  // compute global FEs
  array_t<real_t> f = normalized_f(f_input, data);
  real_t current_objective = quiet_nan<real_t>();
  array_t<real_t> candidate_f, rung_fes;
  real_t onenorm_g = data.active_rungs.size() ? inf<real_t>() : 0;
  real_t epsilon = std::numeric_limits<float>::epsilon();
  size_t globali = 0;

  compute_rung_fes_from_shifts(f, data, rung_fes);
  for (; globali < global_iterations && onenorm_g > _global_solver_tolerance;
       globali++) {
    // fixed-point iteration
    candidate_f = iterate_f(rung_fes, data);
    // compute gradient
    onenorm_g = compute_onenorm(data, f, candidate_f);
    // update rung FEs
    compute_rung_fes_from_shifts(candidate_f, data, rung_fes);
    f = candidate_f;
  }
  tss_log_printf("Finished after %lu iterations with onenorm_g %f\n", globali,
                 onenorm_g);
#ifdef DEBUG_OUTPUT
  std::cout << "Finished after " << globali << " iterations with onenorm_g "
    << onenorm_g << std::endl;
  for (auto wi : data.active_windows) {
    std::cout<<wi<<": "<<f[wi]<<std::endl;
  }
#endif
  return {f, rung_fes, globali, onenorm_g};
}

void free_energy_estimator_t::update_reporting(const graph_t &graph) {
  _expectations = solve_fes_infinite_eta(data, graph);
  _weights = data.p;
}

/*
  Given current guesses for the rung FEs, window weights, rung visit fractions,
  per-window rung weights, and window_fes, computes update guesses for the
  global rung FEs
 */
void free_energy_estimator_t::update_global_estimates(const graph_t &graph) {
  // get per-window weight and FE values
  array_t<real_t> fes;
  std::tie(_f, fes, _last_solve_iterations, _last_solve_error) =
      improve_fes(graph, _f, _solver_iterations);

  // compute window weights incorporating second feedback
  auto &log_weights = eigen_memory_pool_t::get_real(data.window_rung_count);
  fill_with_window_rung_counts(start, data.active_windows);
  real_t eta_over_one_plus_eta = _visit_control_eta / (1 + _visit_control_eta);
  window_list_execute(data.active_windows,
                      [&](auto &window, size_t wi, size_t awi) {
                        auto &lw = window._log_weights;
                        lw = data.log_window_weights[wi];
                        const auto &lv = window._log_rung_volumes;
                        auto num_rungs = window.num_rungs();

                        // compute unnormalized pi
                        window.rung_execute([&](size_t ri, size_t rung) {
                          if (std::isfinite(lv[ri])) {
                            lw[ri] += eta_over_one_plus_eta *
                                      (fes[rung] - data.window_fes[wi][ri]);
                          }
                        });

                        // handle inf
                        if ((lw == inf<real_t>()).any()) {
                          lw = (lw == inf<real_t>())
                            .select(array_t<real_t>::Zero(num_rungs),
                                           -inf<real_t>());
                        }

                         // insert weights into the batch norm computation
                        log_weights.segment(start[awi], num_rungs) = lw;
                      });

  // batch norm computation of weights in each window together for efficiency
  log_sum_exp(start, log_weights, log_denoms);

  // normalize, regularize, and check
  window_list_execute(data.active_windows,
                      [&](auto &window, size_t wi, size_t awi) {
                        auto &lw = window._log_weights;
                        lw = componentwise_log_sum_exp(
                            lw + _log_one_minus_epsilon_pi - log_denoms[awi],
                            _log_epsilon_pi + data.log_window_weights[wi]);
                        check_finite_log(lw, "weight is not finite");
                      });

  window_execute([&](auto &window, size_t wi) {
    window._weights = data.window_weights[wi];
    // ensure that we don't try to do moves in windows we haven't solved for
    if (data.window_map[wi] == -1) {
      window._initialized = false;
    }
  });
}

void free_energy_estimator_t::get_importance_ratios(
    const std::vector<std::unordered_map<size_t, real_t>> &replica_energies,
    const std::vector<size_t> &replica_windows) {
  // we don't actually have to care whether the number of replicas stays the
  // same over time!
  size_t replica_count = replica_windows.size();
  assert(replica_energies.size() == replica_count);

  // batch together the log(sum(exp)) computations from all active windows to
  // let vectorization work more effectively
  fill_with_window_rung_counts(start, replica_windows);
  for (auto &window : _windows) {
    window.clear_ratios(replica_count);
  }
  array_t<real_t> all_actions(start.back());
  for (size_t r = 0; r < replica_count; r++) {
    auto actions =
        _windows[replica_windows[r]].compute_actions(replica_energies[r]);
    all_actions.segment(start[r], actions.size()) = actions;
  }
  log_sum_exp(start, all_actions, log_denoms);

  // compute importance ratios for each replica
  for (size_t r = 0; r < replica_count; r++) {
    _windows[replica_windows[r]].compute_importance_ratios(
        replica_energies[r], r, log_denoms[r]);
  }
}

array_t<real_t> free_energy_estimator_t::unified_detcov() const {
  array_t<real_t> detcovs;
  detcovs.setConstant(_rung_count, real_t());
  array_t<real_t> detcov_weights;
  detcov_weights.setConstant(_rung_count, real_t());

  window_execute([&](const auto &window, size_t /*wi*/) {
    size_t window_weight = window.total_count();
    auto window_detcov = window.estimated_detcov();
    window.rung_execute([&](size_t ri, size_t rung) {
      detcovs[rung] += window_weight * window_detcov[ri];
      detcov_weights[rung] += window_weight;
    });
  });
  return detcovs / detcov_weights;
}

array_t<real_t> free_energy_estimator_t::unified_tilts() const {
  array_t<real_t> tilts;
  tilts.setConstant(_rung_count, real_t());
  array_t<real_t> tilt_weights;
  tilt_weights.setConstant(_rung_count, real_t());

  window_execute([&](const auto &window, size_t /*wi*/) {
    size_t window_weight = window.total_count();
    auto window_tilts = window.overall_tilt();
    window.rung_execute([&](size_t ri, size_t rung) {
      tilts[rung] += window_weight * window_tilts[ri];
      tilt_weights[rung] += window_weight;
    });
  });
  return tilts / tilt_weights;
}

void free_energy_estimator_t::print() {
  printf("FREE ENERGIES\n");
  for (Eigen::Index ei = 0; ei < _rung_count; ei++) {
    printf("%.10f\n", _expectations[ei]);
  }
  printf("ERRORS\n");
  for (Eigen::Index ei = 0; ei < _rung_count; ei++) {
    printf("%.10f\n", _rung_errors[ei]);
  }
  for (const auto &pair_error : _pair_errors) {
    printf("%lu to %lu: %f\n", pair_error.first.first, pair_error.first.second,
           pair_error.second);
  }
  for (size_t w = 0; w < _windows.size(); w++) {
    const auto &window = _windows[w];
    printf("Window %lu\n", w);
    for (size_t ei = 0; ei < window.num_rungs(); ei++) {
      printf("%.10f\n", window._expectations[ei]);
    }
    printf("log(weights)\n");
    for (size_t ei = 0; ei < window.num_rungs(); ei++) {
      printf("%.10f\n", window._log_weights[ei]);
    }
    printf("tilts\n");
    array_t<real_t> tilt = window.overall_tilt();
    for (size_t ei = 0; ei < window.num_rungs(); ei++) {
      printf("%.10f\n", tilt[ei]);
    }
  }
}
