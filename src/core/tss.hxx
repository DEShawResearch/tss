#pragma once

#include "communicator.hxx"
#include "epochs.hxx"
#include "free_energy_estimator.hxx"
#include "graph.hxx"
#include "replica.hxx"
#include "visits.hxx"

class communicator_t;
class estimator_t;

class tss {
  std::set<std::string> _observable_names;

  template <typename epoch_function_t>
  void history_execute(epoch_function_t &&f) {
    f(_visits);
    f(_free_energies);
  }
public:
  // stride at which the estimators are actually updated
  size_t _moves_per_estimator_update;
  size_t _move_counter;
  epoch_t _epochs;
  std::shared_ptr<communicator_t> _comm;
  free_energy_estimator_t _free_energies;
  graph_t _graph;
  std::vector<replica_t> _replicas;
  visits_t _visits;
  std::unordered_map<std::string,
                     std::vector<std::unordered_map<size_t, real_t>>>
      _replica_rung_observables;
  bool _compute_errors;
  bool _populated;

  // caching
  array_t<real_t> _expectations;
  array_t<real_t> _weights;
  pair_to_value_map_t _pair_errors;

  void
  init(const graph_t &graph, size_t replica_count,
       const std::vector<std::shared_ptr<estimator_t>> &estimator_factories,
       std::shared_ptr<communicator_t> comm);
  std::vector<std::shared_ptr<state_t>>
  populate_replicas(const std::vector<std::string> &starting_rungs,
                    const std::vector<std::shared_ptr<state_t>> &states);

  // Methods

  void reinit();
  void cache_output_fields();
  void sample(move_types::move_base_t *mover);
  bool update_tss_state();
  void move(move_types::move_base_t *mover);
  void step(move_types::move_base_t *mover);
  void update_estimates();
  void update_reporting();
  void update_errors();
  void stop(bool suppress_output=false);
  size_t time();

  // Map functions

  void update_state() {
    local_replica_execute(
        [&](auto &r, size_t /*ri*/) { r.update_state(_graph); });
  }

  template <typename replica_function_t>
  void all_replica_execute(replica_function_t &&f) {
    for (size_t ri = 0; ri < _replicas.size(); ri++) {
      f(_replicas[ri], ri);
    }
  }

  template <typename replica_function_t>
  void remote_replica_execute(replica_function_t &&f) {
    all_replica_execute([&](auto &r, size_t ri) {
      if (!_comm->is_local_replica(ri)) {
        f(r, ri);
      }
    });
  }

  template <typename replica_function_t>
  void local_replica_execute(replica_function_t &&f) {
    all_replica_execute([&](auto &r, size_t ri) {
      if (_comm->is_local_replica(ri)) {
        f(r, ri);
      }
    });
  }
};
