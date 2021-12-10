#pragma once

#include "../util/util.hxx"
#include "history.hxx"

class graph_t;
class window_t;

struct estimator_type_t {
  static std::string detcov() { return "6948aa8f"; }
};

class estimator_t
    : public std::enable_shared_from_this<estimator_t> {
public:
  virtual ~estimator_t() {}

  virtual void init(const std::vector<size_t> & /*rungs*/) = 0;

  virtual void add_epoch() {}
  virtual void drop_epoch() {}
  virtual void update(
      const graph_t & /*graph*/, const window_t & /*free_energies*/,
      const std::vector<array_t<real_t>> & /*replica_log_ratios*/,
      const std::vector<size_t> & /*replicas*/,
      const std::unordered_map<std::string,
                               std::vector<std::unordered_map<size_t, real_t>>>
          & /*replica_observables*/
  ) {}
  virtual void print() {}

  virtual array_t<real_t> expectations() const = 0;
  virtual array_t<real_t> estimate(const window_t &free_energies,
                                   size_t exclude_epoch = -1) const = 0;
  virtual std::shared_ptr<estimator_t> create() const = 0;
  virtual std::string type() const = 0;
  virtual std::vector<std::string> required_observables() { return {}; }
};
