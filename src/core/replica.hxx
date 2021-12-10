#pragma once

#include "state.hxx"

class graph_t;
class window_t;
namespace move_types {
  class move_base_t;
}

class replica_t {
public:
  size_t _last_window_index;
  size_t _window_index;
  size_t _last_rung;
  size_t _rung;
  array_t<real_t> _offset;
  bool _serialize_state;
  std::shared_ptr<state_t> _state;
  std::map<std::string, std::unordered_map<size_t, real_t>> _observables;
  real_t _unable_to_move;

  replica_t() : _serialize_state(true), _unable_to_move(false) {
    _offset.setConstant(1, 0);
  }

  void init(std::shared_ptr<state_t> state) {
    _state = state;
  }

  // Methods

  void set_window_rung(const graph_t &graph, size_t window_index, size_t rung);

  array_t<real_t> log_probability_mass(
      const window_t &window,
      const std::unordered_map<size_t, real_t> &energies) const;

  void update_lambda(const graph_t &graph, move_types::move_base_t &mover,
                     const window_t &window,
                     const std::unordered_map<size_t, real_t> &energies);

  void update_window(const graph_t &graph, move_types::move_base_t &mover);

  void update_state(const graph_t &graph);

  void update_observables(const std::set<std::string> &observable_names,
                          const graph_t &graph);

  void clear_observables(const graph_t &graph);

  // Serialization

  std::vector<size_t> pack_state() const {
    return {_rung, size_t(_unable_to_move)};
  }

  void unpack_state(const std::vector<size_t>& wrung) {
    _last_rung = _rung;
    _rung = wrung[0];
    _unable_to_move = wrung[1];
  }
};
