#pragma once

#include "../util/eigen.hxx"

class replica_t;

struct communicator_t : public std::enable_shared_from_this<communicator_t> {
  virtual ~communicator_t() {}
  virtual void init(size_t /*replica_count*/) {}
  virtual void start() {}
  virtual void stop() {}

  // communication
  virtual void send_state(const size_t, const replica_t &) {
    throw std::runtime_error("send_state not defined");
  }
  virtual void receive_state(replica_t &) {
    throw std::runtime_error("receive_state not defined");
  }
  virtual void
  send_observable(const std::unordered_map<size_t, real_t> & /*observable*/) {
    throw std::runtime_error("send_observable not defined");
  }
  virtual void
  receive_observable(const size_t /*r_index*/,
                     std::unordered_map<size_t, real_t> & /*observable*/) {
    throw std::runtime_error("receive_observable not defined");
  }

  virtual bool is_local_replica(size_t rank) const = 0;
  virtual bool is_rank_zero() const = 0;
};

struct local_comm_t : public communicator_t {
  bool is_local_replica(size_t /*rank*/) const override;
  bool is_rank_zero() const override;
};
