#pragma once

#include "window.hxx"

class coordinate_weighting_t;
class epoch_t;
class graph_t;
class visits_t;

struct windows_data_t {
  // rung weights and ln(rung weights) for each window
  std::vector<array_t<real_t>> window_weights;
  std::vector<array_t<real_t>> log_window_weights;

  // tilts for each window
  std::vector<array_t<real_t>> tilts;

  // rung FEs for each window
  std::vector<array_t<real_t>> window_fes;
  // either empirical or estimated count fractions for each rung
  array_t<real_t> p;
  array_t<real_t> log_p;
  // either empirical or estimated weights for each window
  array_t<real_t> pi;
  array_t<real_t> log_pi;
  // list of windows which are currently active (have nontrivial FE information)
  std::vector<size_t> active_windows;
  // map from overall window index to index in active set
  std::vector<int> window_map;
  // number of active windows at each rung.  Storing as real_t for ease of use.
  array_t<real_t> rung_window_count;
  // flag whether each rung is active (for use in normalization and estimation)
  std::vector<bool> active_rung_mask;
  // list of active rungs, corresponding to the active rung mask
  std::vector<size_t> active_rungs;
  // sum of rungs in all active windows
  size_t window_rung_count;

  real_t eta_plus_one;
};

class free_energy_estimator_t {
  // temporaries
  mutable array_t<real_t> log_denoms;
  mutable std::vector<size_t> start;
  mutable std::vector<size_t> rung_visited;
  mutable windows_data_t data;

public:
  //
  // Data members
  //

  // main data contents
  std::vector<window_t> _windows;
  real_t _rung_count;

  // updated FE-related quantities: window shifts and overall FEs
  array_t<real_t> _f;
  array_t<real_t> _expectations;

  // estimated weights and associated parameters
  real_t _weight_epsilon;
  array_t<real_t> _weights;
  real_t _log_epsilon_pi;
  real_t _log_one_minus_epsilon_pi;

  // error estimates and indices
  pair_to_value_map_t _pair_errors;
  array_t<real_t> _rung_errors;

  // visit control strength.  0 disables.
  real_t _visit_control_eta;

  // global solve parameters
  size_t _solver_iterations;
  real_t _global_solver_tolerance;

  // global solve results
  size_t _last_solve_iterations;
  real_t _last_solve_error;

  bool _omit_solve;

  //
  // Methods
  //

  // needed for deserialization
  free_energy_estimator_t() {}

  // configuration
  void
  init(const graph_t &graph,
       const std::vector<std::shared_ptr<estimator_t>> &estimator_factories);
  void
  register_pair_error_estimate(const std::pair<size_t, size_t> &rung_pair) {
    _pair_errors[rung_pair] = std::numeric_limits<real_t>::infinity();
  }

  // epoch handling
  void add_epoch();
  void drop_epoch();

  // updates to estimated quantities
  void update_estimates(
      const graph_t &graph,
      const std::unordered_map<std::string,
                               std::vector<std::unordered_map<size_t, real_t>>>
          &replica_observables,
      const std::vector<size_t> &replica_windows);
  void update_tilts(const std::unordered_map<size_t, std::vector<size_t>>
                        &window_rung_visits);
  void update_global_estimates(const graph_t &graph);
  void update_reporting(const graph_t &graph);
  void update_errors(const graph_t &graph, const epoch_t &epochs);

  // data gathering
  void get_window_values(windows_data_t &data, real_t eta,
                         const size_t exclude_epoch = -1);
  void get_active_windows(windows_data_t &data, const graph_t &graph) const;
  void get_window_data(windows_data_t &data, const graph_t &graph, real_t eta,
                       const size_t exclude_epoch = -1);
  void get_rung_fractions(windows_data_t &data);
  void get_importance_ratios(
      const std::vector<std::unordered_map<size_t, real_t>> &replica_energies,
      const std::vector<size_t> &replica_windows);
  array_t<real_t> solve_fes_infinite_eta(windows_data_t &data,
                                         const graph_t &graph,
                                         size_t exclude_epoch = -1);
  void window_weights_qr(windows_data_t &data, const graph_t &graph);

  // linear equation solving to get window weights
  void global_weights(windows_data_t &data);

  // nonlinear equation solving to get window shifts
  void compute_rung_fes_from_shifts(const array_t<real_t> &f,
                                    const windows_data_t &data,
                                    array_t<real_t> &computed_rung_fes);
  array_t<real_t> iterate_f(const array_t<real_t> &rung_fes,
                                const windows_data_t &data);
  std::tuple<array_t<real_t>, array_t<real_t>, size_t, real_t>
  improve_fes(const graph_t &graph, const array_t<real_t> &f_input,
              size_t global_iterations);

  // output
  array_t<real_t> unified_detcov() const;
  array_t<real_t> unified_tilts() const;
  void print();

  // map functions
  template <typename window_function_t>
  void window_execute(window_function_t &&f) {
    for (size_t wi = 0; wi < _windows.size(); wi++) {
      f(_windows[wi], wi);
    }
  }

  template <typename window_function_t>
  void window_execute(window_function_t &&f) const {
    for (size_t wi = 0; wi < _windows.size(); wi++) {
      f(_windows[wi], wi);
    }
  }

  template <typename window_function_t>
  void window_list_execute(const std::vector<size_t> &window_indices,
                           window_function_t &&f) {
    for (size_t awi = 0; awi < window_indices.size(); awi++) {
      auto wi = window_indices[awi];
      f(_windows[wi], wi, awi);
    }
  }

  void fill_with_window_rung_counts(std::vector<size_t> &indices,
                                    const std::vector<size_t> &window_indices) {
    auto window_count = window_indices.size();
    indices.resize(window_count + 1);
    indices[0] = 0;
    window_list_execute(window_indices,
                        [&](auto &window, size_t /*wi*/, size_t awi) {
                          indices[awi + 1] = indices[awi] + window.num_rungs();
                        });
  }
};
