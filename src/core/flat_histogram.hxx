#pragma once

#include "estimator.hxx"
#include "graph.hxx"

class flat_distribution_t : public estimator_t {
  array_t<real_t> _detcov;

public:
  std::string type() const { return estimator_type_t::detcov(); }

  void init(const std::vector<size_t> &rungs) override {
    _detcov.setConstant(rungs.size(), 1);
  }

  array_t<real_t> expectations() const override { return _detcov; }

  array_t<real_t> estimate(const window_t & /*free_energies*/,
                           size_t /*exclude_epoch = -1*/) const override {
    return expectations();
  }

  std::shared_ptr<estimator_t> create() const {
    return std::make_shared<flat_distribution_t>();
  }
};
