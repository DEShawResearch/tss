#pragma once

#include <algorithm>
#include <deque>
#include <set>
#include "../util/eigen.hxx"

class graph_t;

class visits_t
{
  std::deque<array_t<size_t>> _rung_visits;
  size_t _rung_count;
  array_t<size_t> _total_visits;

  void compute_total_visits();
public:
  void init(size_t rung_count);

  void visit(const size_t rung);

  void add_epoch();

  void drop_epoch();

  void print() const;

  size_t visit_count(size_t rung, size_t exclude_epoch) const;

  size_t visit_count(size_t rung) const {
    return _total_visits[rung];
  }

  array_t<size_t> visit_counts() const {
    return _total_visits;
  }

  array_t<size_t> visit_counts(size_t exclude_epoch) const;
};
