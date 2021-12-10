#include "eigen.hxx"

eigen_memory_pool_t& eigen_memory_pool_t::singleton() {
  static eigen_memory_pool_t single_object;
  return single_object;
}
