#include <cassert>
#include "graph.hxx"
#include "visits.hxx"

void visits_t::init(size_t rung_count) {
  _rung_count = rung_count;
  _total_visits.setConstant(rung_count, 0);
  _rung_visits.clear();_rung_visits.resize(1);
  _rung_visits[0].setConstant(rung_count, 0);
}

void visits_t::visit(const size_t rung) {
  // record a visit to a rung
  _rung_visits.front()[rung]++;
  _total_visits[rung]++;
}

void visits_t::add_epoch() {
  // generate new epochs as appropriate
  _rung_visits.push_front(array_t<size_t>::Constant(_rung_count, 0));
}

void visits_t::compute_total_visits() {
  _total_visits.fill(0);
  for (auto &epoch_visits : _rung_visits) {
    for (size_t rung = 0; rung < _rung_count; rung++) {
      _total_visits[rung] += epoch_visits[rung];
    }
  }
}

void visits_t::drop_epoch() {
  _rung_visits.pop_back();
  // re-build _total_visits with the remaining epochs
  compute_total_visits();
}

void visits_t::print() const {
  printf("COUNTS\n");
  for (size_t vi = 0; vi < _rung_count; vi++) {
    printf("%lu\n", _total_visits[vi]);
  }
}

size_t visits_t::visit_count(size_t rung, size_t exclude_epoch) const {
  assert(exclude_epoch < _rung_visits.size());
  return visit_count(rung) - _rung_visits[exclude_epoch][rung];
}

array_t<size_t> visits_t::visit_counts(size_t exclude_epoch) const {
  assert(exclude_epoch < _rung_visits.size());
  return visit_counts() - _rung_visits[exclude_epoch];
}
