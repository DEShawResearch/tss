#include "communicator.hxx"

bool local_comm_t::is_local_replica(size_t /*rank*/) const {
  return true;
}

bool local_comm_t::is_rank_zero() const {
  return true;
}
