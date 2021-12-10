#pragma once

#include "estimator.hxx"
#include "../util/eigen.hxx"

class covariance_estimator_t : public estimator_t {
  history_t<vector_t> _derivatives;
  history_t<matrix_t> _ddTs;
  array_t<real_t> _detcov_weight;

public:
  std::string type() const { return estimator_type_t::detcov(); }

  void init(const std::vector<size_t> &rungs) override;

  void add_epoch() override;

  void drop_epoch() override;

  template <typename value_t>
  void update_epoch_estimates(
      const graph_t &graph, const window_t &free_energies,
      const std::vector<size_t> &replicas,
      const std::vector<array_t<real_t>> &replica_log_ratios,
      const std::vector<std::vector<value_t>> &values,
      history_t<value_t> &estimator);

  void update(const graph_t &graph, const window_t &free_energies,
              const std::vector<array_t<real_t>> &replica_log_ratios,
              const std::vector<size_t> &replicas,
              const std::unordered_map<
                  std::string, std::vector<std::unordered_map<size_t, real_t>>>
                  &replica_observables) override;

  array_t<real_t> estimate(const window_t &free_energies,
                           size_t exclude_epoch = -1) const override;

  void print();

  array_t<real_t> expectations() const override {
    return _detcov_weight;
  }

  std::shared_ptr<estimator_t> create() const {
    return std::make_shared<covariance_estimator_t>();
  }

  std::vector<std::string> required_observables() override {
    return {"energy"};
  }
};
