#include "distributions.hxx"

rng_manager_t& rng_manager_t::singleton() {
  static rng_manager_t single_object;
  return single_object;
}
