#include "../util/eigen.hxx"
#include "graph.hxx"
#include "move_types.hxx"
#include "replica.hxx"
#include "window.hxx"

void replica_t::set_window_rung(const graph_t &graph, size_t window_index,
                                size_t rung) {
  _window_index = _last_window_index = window_index;
  _rung = _last_rung = rung;
  _offset.setConstant(graph.rung_dimension(rung), real_t());
}

array_t<real_t> replica_t::log_probability_mass(
    const window_t &window,
    const std::unordered_map<size_t, real_t> &energies) const {
  const auto &free_energy = window._expectations;
  const auto &log_weights = window._log_weights;
  auto rung_count = window.num_rungs();

  array_t<real_t> local_energy(rung_count);
  tss_log_printf("Move weights\n");
  window.rung_execute([&](size_t ri, size_t rung) {
    local_energy[ri] = free_energy[ri] - energies.at(rung) + log_weights[ri];
    tss_log_printf("Rung %lu: %f (%f, %f, %f)\n", rung, local_energy[ri],
                   free_energy[ri], energies.at(rung), log_weights[ri]);
  });
  return local_energy;
}

void replica_t::update_lambda(
    const graph_t &graph, move_types::move_base_t &mover,
    const window_t &window,
    const std::unordered_map<size_t, real_t> &energies) {
  _last_rung = _rung;
  _unable_to_move = false;

  // only do a move if the window has FE information
  if (!window.initialized()) {
    _unable_to_move = true;
    return;
  }
  // get the log probability masses
  auto log_p = log_probability_mass(window, energies);

  // do the move
  std::tie(_rung, _offset) =
      mover.move(graph, _window_index, window._rungs, log_p, _rung, _offset);
  tss_log_printf("From %lu to %lu\n", _last_rung, _rung);

  check_finite(_offset, "lambda resulting from move was nonfinite");
}

void replica_t::update_window(const graph_t &graph,
                              move_types::move_base_t &mover) {
  if (_unable_to_move) {
    return;
  }
  _last_window_index = _window_index;
  _window_index = mover.window_move(graph, _window_index, _rung);
  tss_log_printf("Window from %lu to %lu\n", _last_window_index, _window_index);
}

void replica_t::update_state(const graph_t &graph) {
  _state->update(graph.rung_to_schedules(_rung), _rung);
}

void replica_t::update_observables(
    const std::set<std::string> &observable_names, const graph_t &graph) {
  // get observables from state
  const auto &schedules = graph.window_schedules[_window_index];
  const auto &evaluation_rungs = graph.window_evaluation_rungs[_window_index];
  auto observables = _state->observables(schedules, evaluation_rungs);
  for (const auto & observable_name : observable_names) {
    auto &_observable = _observables[observable_name];
    const auto &observable = observables[observable_name];
    for (size_t i = 0; i < evaluation_rungs.size(); i++) {
      _observable[evaluation_rungs[i]] = observable[i];
    }
  }
}

void replica_t::clear_observables(const graph_t &graph) {
  const auto &evaluation_rungs = graph.window_evaluation_rungs[_window_index];
  // write NaNs into all rung observables corresponding to the current window
  for (auto &observable : _observables) {
    observable.second.clear();
  }
}
