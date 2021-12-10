#pragma once

#include "../util/randgen.h"
#include "graph.hxx"
#include "../util/eigen.hxx"
#include "../util/util.hxx"

namespace move_types {
inline real_t evaluate_cdf(const array_t<real_t> &pdf, lambda_t l) {
  real_t p = 0;
  for (size_t rung = 0; rung < l.first; rung++) {
    p += pdf[rung];
  }
  return p + pdf[l.first] * (l.second + 0.5);
}

inline lambda_t invert_cdf(const array_t<real_t> &pdf, real_t p) {
  size_t rung = 0;
  real_t sum = 0;
  for (; p > sum + pdf[rung]; rung++) {
    sum += pdf[rung];
  }
  // clamp
  real_t fraction = std::min(pdf[rung], std::max(real_t(), p - sum));
  return lambda_t(rung, fraction / pdf[rung] - 0.5);
}

struct move_base_t {
  randgen _rng;

  move_base_t(uint64_t seed = 0) { _rng.seed(seed); }

  virtual lambda_t operator()(const array_t<real_t> & /*p*/,
                              lambda_t current_lambda) {
    return current_lambda;
  }

  virtual size_t window_move(const graph_t &graph, size_t window_index,
                             size_t rung) {
    return graph.swap_window(window_index, rung);
  }

  virtual std::pair<size_t, array_t<real_t>>
  move(const graph_t &graph, const size_t window_index,
       const std::vector<size_t> &rungs, const array_t<real_t> &log_p,
       size_t rung, array_t<real_t> offset) {
    // If window dimension is zero, the window is unstructured and we must do an
    // independent move.  Otherwise, do whatever mover dictates in each
    // dimension.
    const auto& move_groups = graph.window_move_groups[window_index];
    size_t window_dimension = move_groups.size();
    if (window_dimension == 0) {
      array_t<real_t> log_p_copy(log_p);
      subtract_maximum_value(log_p_copy);
      auto ind = independent_move(log_p_copy.exp());
      size_t rung = rungs[ind.first];
      // set random offset within rung
      size_t dim = graph.rung_dimension(rung);
      array_t<real_t> offset(dim);
      for (size_t d = 0; d < dim; d++) {
        offset[d] = random() - 0.5;
      }
      return {rung, offset};
    }
    // ensure that offset is correctly sized
    if (offset.size() != Eigen::Index(window_dimension)) {
      throw std::invalid_argument("Offset size must equal window dimension");
    }

    // permute dimensions
    std::vector<size_t> dimensions(window_dimension);
    std::iota(dimensions.begin(), dimensions.end(), 0);
    std::random_shuffle(dimensions.begin(), dimensions.end(),
                        [&](int ct) { return random() * ct; });

    const auto& index_map = graph.window_rung_to_ri[window_index];
    // move in each dimension
    for (size_t d = 0; d < window_dimension; d++) {
      size_t dim = dimensions[d];
      const auto &move_group =
          graph.rung_move_groups[move_groups[dim].at(rung)];
      // intersect this with the trusted rung set and map p
      size_t group_size = move_group.size();
      std::vector<size_t> rung_subset(group_size);
      array_t<real_t> log_p_subset(group_size);
      size_t group_index = 0;
      for (auto group_rung : move_group) {
        rung_subset[group_index] = group_rung;
        log_p_subset[group_index] = log_p[index_map.at(group_rung)];
        group_index++;
      }

      // note that this is not properly normalized.  It doesn't need to be.
      subtract_maximum_value(log_p_subset);

      // map rung into active subset, do move, map back
      size_t rung_index =
          std::find(rung_subset.begin(), rung_subset.end(), rung) -
          rung_subset.begin();
      std::tie(rung_index, offset[dim]) =
          (*this)(log_p_subset.exp(), {rung_index, offset[dim]});
      if (std::abs(offset[dim]) > 0.5) {
        throw std::out_of_range("lambda must be between -0.5 and 0.5");
      }
      rung = rung_subset[rung_index];
    }
    return {rung, offset};
  }

  double random() { return _rng.yield().as_double_co(); }

  virtual lambda_t independent_move(const array_t<real_t> &p) {
    real_t urandom = random() * p.sum();
    return invert_cdf(p, urandom);
  }
};

struct uniform_move_t : public move_base_t {
  uniform_move_t(uint64_t seed = 0) : move_base_t(seed) {}

  lambda_t operator()(const array_t<real_t> &p,
                      lambda_t /*current_rung*/) override {
    real_t uniform_random = random() * p.size();
    return lambda_t(std::floor(uniform_random), 0);
  }
};

struct independent_move_t : public move_base_t {
  independent_move_t(uint64_t seed = 0) : move_base_t(seed) {}

  std::pair<size_t, array_t<real_t>>
  move(const graph_t &graph, const size_t /*window_index*/,
       const std::vector<size_t> &rungs, const array_t<real_t> &log_p,
       size_t rung, array_t<real_t> /*offset*/) override {
    array_t<real_t> log_p_copy(log_p);
    subtract_maximum_value(log_p_copy);
    auto ind = independent_move(log_p_copy.exp());
    size_t new_rung = rungs[ind.first];
    // set random offset within rung
    size_t dim = graph.rung_dimension(rung);
    array_t<real_t> new_offset(dim);
    for (size_t d = 0; d < dim; d++) {
      new_offset[d] = random() - 0.5;
    }
    return {new_rung, new_offset};
  }

  lambda_t operator()(const array_t<real_t> &p,
                      lambda_t /*current_rung*/) override {
    return independent_move(p);
  }
};
} // namespace move_types
