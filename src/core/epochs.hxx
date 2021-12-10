#pragma once

#include "../util/util.hxx"
#include <deque>

class epoch_t {
  std::deque<size_t> _epoch_times; // epoch end times
  size_t _time;
  size_t _earliest_epoch_start;
  double _relative_epoch_size; // phi
  double _alpha;

public:

  epoch_t() {
    _epoch_times.resize(1);
    _epoch_times[0] = 0;
    _earliest_epoch_start = 0;
    _time = 1;
    configure(0.5, 16);
  }

  void configure(real_t alpha, size_t number_of_epochs) {
    _alpha = alpha;
    _relative_epoch_size =
        (alpha == 0) ? 2 : std::pow(alpha, -1.0 / number_of_epochs);
    if (alpha == 0) {
      _epoch_times[0] = size_t(-1);
    }
  }

  void increment_time() {
    _time++;
  }

  bool drop_epoch() {
    if (_alpha * _time - _alpha > 0 && _epoch_times[0] < _alpha * _time) {
      _earliest_epoch_start = _epoch_times[0];
      _epoch_times.pop_front();
      return true;
    }
    return false;
  }

  bool epoch_full() {
    return _time >= _epoch_times.back();
  }

  bool add_epoch() {
    if (_time <= _epoch_times.back()) {
      // don't need to allocate space for next epoch yet
      return false;
    }
    // generate new epochs as appropriate
    size_t next_end_time = ceil(_relative_epoch_size * _epoch_times.back());
    _epoch_times.push_back(std::max(_epoch_times.back() + 1, next_end_time));
    return true;
  }

  size_t time() const { return _time; }

  size_t count() const { return _epoch_times.size(); }

  std::vector<real_t> weights() const {
    std::vector<real_t> epoch_weight(_epoch_times.size());
    real_t inv_span = 1 / (real_t)(_time - _earliest_epoch_start);
    // note that the order of epoch weights is reversed from that of epoch times
    auto time_iter = _epoch_times.rbegin() + 1;
    auto weight_iter = epoch_weight.begin();
    size_t last_time = _time;
    for (; time_iter != _epoch_times.rend(); time_iter++, weight_iter++) {
      *weight_iter = (last_time - *time_iter) * inv_span;
      last_time = *time_iter;
    }
    epoch_weight.back() = (_epoch_times[0] - _earliest_epoch_start) * inv_span;
    return epoch_weight;
  }
};
