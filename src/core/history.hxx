#pragma once

#include "../util/util.hxx"
#include "../util/eigen.hxx"
#include <deque>

template <class value_t> class history_t {
protected:
  std::deque<array_t<value_t>> _estimate_history;
  std::deque<size_t> _count;
  value_t _default_value;

private:
  friend class tss;

public:
  array_t<value_t> _expectations;
  std::vector<size_t> _rungs;

  history_t(value_t default_value = value_t())
      : _default_value(default_value) {}

  // accessors
  value_t estimate(size_t epoch, size_t ri) const {
    return _estimate_history.rbegin()[epoch][ri];
  }

  value_t &estimate(size_t epoch, size_t ri) {
    return _estimate_history.rbegin()[epoch][ri];
  }

  const array_t<value_t> &estimates(size_t epoch) const {
    return _estimate_history.rbegin()[epoch];
  }

  array_t<value_t> &estimates(size_t epoch) {
    return _estimate_history.rbegin()[epoch];
  }

  size_t count(size_t epoch) const { return _count.rbegin()[epoch]; }

  size_t &count(size_t epoch) { return _count.rbegin()[epoch]; }

  size_t total_count() const {
    return std::accumulate(_count.cbegin(), _count.cend(), size_t());
  }

  size_t total_count(size_t exclude_epoch) const {
    size_t c = 0;
    for (size_t epoch = 0; epoch < num_epochs(); epoch++) {
      if (epoch != exclude_epoch) {
        c += count(epoch);
      }
    }
    return c;
  }

  value_t operator()(size_t ri) const { return _expectations[ri]; }

  value_t &operator()(size_t ri) { return _expectations[ri]; }

  size_t num_epochs() const { return _estimate_history.size(); }

  size_t num_rungs() const { return _rungs.size(); }

  // methods
  void init(const std::vector<size_t> &rungs);

  void add_epoch();

  void drop_epoch();

  void print();

  template <typename rung_function_t>
  void rung_execute(rung_function_t &&f) const {
    for (size_t ri = 0; ri < _rungs.size(); ri++) {
      f(ri, _rungs[ri]);
    }
  }
};
