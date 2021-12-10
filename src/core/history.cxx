#include "history.hxx"
#include "../util/eigen.hxx"

template <class value_t>
void history_t<value_t>::init(const std::vector<size_t> &rungs) {
  _rungs = rungs;
  _estimate_history.clear();
  _count.clear();
  _expectations = array_t<value_t>::Constant(num_rungs(), _default_value);
  add_epoch();
}

template <class value_t> void history_t<value_t>::add_epoch() {
  // generate new epochs as appropriate
  _estimate_history.push_back(
      array_t<value_t>::Constant(num_rungs(), _default_value));
  _count.push_back(real_t());
}

template <class value_t> void history_t<value_t>::drop_epoch() {
  _estimate_history.pop_front();
  _count.pop_front();
}

template <class value_t> void history_t<value_t>::print() {}

template class history_t<real_t>;
template class history_t<size_t>;
template class history_t<matrix_t>;
template class history_t<vector_t>;
template class history_t<array_t<real_t>>;
