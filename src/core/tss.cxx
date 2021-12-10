#include "tss.hxx"
#include "communicator.hxx"
#include "estimator.hxx"

void tss::init(
    const graph_t &graph, size_t replica_count,
    const std::vector<std::shared_ptr<estimator_t>> &estimator_factories,
    std::shared_ptr<communicator_t> comm) {
  // do general initialization
  _moves_per_estimator_update = 1;
  _move_counter = 0;
  _graph = graph;
  _comm = comm;
  _comm->init(replica_count);
  _visits.init(_graph.rung_count());
  if (_comm->is_rank_zero()) {
    _free_energies.init(graph, estimator_factories);
  }
  _compute_errors = true;

  // figure out which kinds of observables we're going to need from our
  // states
  _observable_names.insert("energy");
  for (const auto estimator : estimator_factories) {
    for (const auto observable_name : estimator->required_observables()) {
      _observable_names.insert(observable_name);
    }
  }

  // create replicas
  _replicas.resize(replica_count);
  _populated = false;
}

std::vector<std::shared_ptr<state_t>>
tss::populate_replicas(const std::vector<std::string> &starting_rungs,
                       const std::vector<std::shared_ptr<state_t>> &states) {
  if (starting_rungs.size() != _replicas.size()) {
    throw std::invalid_argument("len(starting_rungs) must equal replica count");
  }
  all_replica_execute([&](auto &r, size_t ri) {
    // starting rung/window
    size_t rung = 0;
    if (starting_rungs[ri] != "") {
      rung = _graph.rung_from_rung_name(starting_rungs[ri]);
    }
    size_t window_index = _graph.rung_to_windows[rung][0];
    // configure replica
    r.set_window_rung(_graph, window_index, rung);
    r._serialize_state = true;
  });
  // add states to the local replicas
  std::vector<std::shared_ptr<state_t>> local_states;
  local_replica_execute([&](auto &r, size_t ri) {
    states[ri]->init(ri);
    r.init(states[ri]);
    local_states.push_back(states[ri]);
  });

  _populated = true;
  return local_states;
}

void tss::reinit() { _comm->start(); }

void tss::cache_output_fields() {
  _expectations = _free_energies._expectations;
  _weights = _free_energies._weights;
  _pair_errors = _free_energies._pair_errors;
}

void tss::move(move_types::move_base_t *mover) {
  if (mover) {
    // use energies and current FE estimates to update lambda
    const auto &energies = _replica_rung_observables["energy"];
    all_replica_execute([&](auto &r, size_t ri) {
      r.update_lambda(_graph, *mover, _free_energies._windows[r._window_index],
                      energies[ri]);
    });
  }

  // inform remote replicas what their updated state should be
  remote_replica_execute([&](auto &r, size_t ri) { _comm->send_state(ri, r); });
}

void tss::sample(move_types::move_base_t *mover) {
  for (auto observable_name : _observable_names) {
    _replica_rung_observables[observable_name].resize(_replicas.size());
  }
  // get values of observables for local replicas (energy, possibly others)
  local_replica_execute([&](auto &r, size_t ri) {
    r.update_observables(_observable_names, _graph);
    for (const auto &oname : _observable_names) {
      const auto& observable = r._observables[oname];
      _replica_rung_observables[oname][ri] = observable;

      if (!_comm->is_rank_zero()) {
        // send observable to rank zero
        _comm->send_observable(observable);
      }
    }
    r.clear_observables(_graph);
  });

  // retrieve values of observables from any remote replicas if rank zero
  if (_comm->is_rank_zero()) {
    remote_replica_execute([&](auto & /*r*/, size_t ri) {
      for (const auto &oname : _observable_names) {
        _comm->receive_observable(ri, _replica_rung_observables[oname][ri]);
      }
    });
  }

  // do move if rank zero, otherwise receive move results
  if (_comm->is_rank_zero()) {
    // cache output fields before move
    cache_output_fields();
    move(mover);
  } else {
    // find out what update we need to make to the local replica state
    local_replica_execute(
        [&](auto &r, size_t /*ri*/) { _comm->receive_state(r); });
  }
}

bool tss::update_tss_state() {
  if (_move_counter++ % _moves_per_estimator_update != 0) {
    return false;
  }
  
  _epochs.increment_time();
  if (!_comm->is_rank_zero()) {
    // only the 0th rank process should update the estimators
    return true;
  }

  // handle adding and dropping epochs
  if (_epochs.add_epoch()) {
    history_execute([&](auto &e) { e.add_epoch(); });
  }
  if (_epochs.drop_epoch()) {
    history_execute([&](auto &e) { e.drop_epoch(); });
  }

  // update visit state
  std::unordered_map<size_t, std::vector<size_t>> window_rung_visits;
  all_replica_execute([&](auto &r, size_t /*ri*/) {
    if (!r._unable_to_move) {
      _visits.visit(r._rung);
      window_rung_visits[r._window_index].push_back(
          _graph.window_rung_to_ri[r._window_index][r._rung]);
    }
  });
  _free_energies.update_tilts(window_rung_visits);

  // update the estimates
  update_estimates();

  // compute the error estimates right _before_ we add an epoch!
  if (_compute_errors && _epochs.epoch_full()) {
    update_errors();
  }
  return true;
}

void tss::step(move_types::move_base_t *mover) {
  if (!_populated) {
    throw std::runtime_error(
        "The model must be initialized before running the estimators");
  }
  tss_log_printf("\nStep %lu\n", time());

  // get new {X, lambda} sample
  sample(mover);

  // use it to update the estimators
  bool updated_state = update_tss_state();

  // this must be after the estimators are updated, but must be at the end of
  // the step for Anton's sake
  if (mover) {
    // update the window, but only if we also updated the tss state
    if(updated_state) {
      all_replica_execute(
          [&](auto &r, size_t /*ri*/) { r.update_window(_graph, *mover); });
    }
  }

  // draw new samples
  update_state();
}

void tss::update_estimates() {
  std::vector<size_t> replica_windows(_replicas.size());
  for (size_t i = 0; i < _replicas.size(); i++) {
    replica_windows[i] = _replicas[i]._window_index;
  }
  _free_energies.update_estimates(_graph, _replica_rung_observables,
                                  replica_windows);
}

void tss::update_reporting() {
  _free_energies.update_reporting(_graph);
}

void tss::update_errors() {
  _free_energies.update_errors(_graph, _epochs);
}

void tss::stop(bool suppress_output) {
  if (!suppress_output && _comm->is_rank_zero()) {
    _free_energies.print();
    _visits.print();
  }
  _comm->stop();
}

size_t tss::time() {
  return _move_counter;
}

