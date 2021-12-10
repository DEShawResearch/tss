#pragma once

#include <complex>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>
#include "util.hxx"

template <typename value_t = real_t>
using array_t = Eigen::Array<value_t, Eigen::Dynamic, 1>;
typedef Eigen::Matrix<real_t, Eigen::Dynamic, 1> vector_t;
typedef Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

typedef Eigen::Triplet<real_t> eigen_triplet_t;
// STL containers holding fixed-size Eigen types require an aligned allocator
typedef std::vector<eigen_triplet_t, Eigen::aligned_allocator<eigen_triplet_t>>
    triplet_list_t;
typedef Eigen::SparseMatrix<real_t> sparse_matrix_t;

template <typename T>
void check_finite(const array_t<T> &value,
                  const std::string msg = "not finite") {
  if (!value.isFinite().all()) {
    throw std::out_of_range(msg);
  }
}

template <typename T>
void check_finite_log(const array_t<T> &value,
                      const std::string msg = "not finite") {
  if ((value == inf<T>()).any() || value.isNaN().any()) {
    throw std::out_of_range(msg);
  }
}

inline real_t determinant(real_t s) {
  return s;
}

inline real_t determinant(const matrix_t& m) {
  return m.determinant();
}

template <typename T>
std::vector<T> stdvector_from_array(const array_t<T>& array) {
  return std::vector<T>(array.data(), array.data() + array.size());
}

struct eigen_memory_pool_t {
  std::unordered_map<Eigen::Index, array_t<float>> float_memory_pool;
  std::unordered_map<Eigen::Index, array_t<float>> second_float_memory_pool;
  std::unordered_map<Eigen::Index, array_t<real_t>> real_memory_pool;
  std::unordered_map<Eigen::Index, array_t<real_t>> second_real_memory_pool;

  template <typename T>
  static array_t<T>&
  get_from_pool(std::unordered_map<Eigen::Index, array_t<T>> &pool,
                Eigen::Index size) {
    if (pool.find(size) == pool.end()) {
      pool[size].resize(size);
    }
    return pool.at(size);
  }

  static array_t<float>& get_float(Eigen::Index size) {
    return get_from_pool(singleton().float_memory_pool, size);
  }

  static array_t<float>& get_second_float(Eigen::Index size) {
    return get_from_pool(singleton().second_float_memory_pool, size);
  }

  static array_t<real_t>& get_real(Eigen::Index size) {
    return get_from_pool(singleton().real_memory_pool, size);
  }

  static array_t<real_t>& get_second_real(Eigen::Index size) {
    return get_from_pool(singleton().second_real_memory_pool, size);
  }

  static eigen_memory_pool_t& singleton();
};

template <typename T, typename T2>
void sum(const std::vector<size_t> &row_start, const array_t<T> &summand,
         array_t<T2> &sums) {
  Eigen::Index count = row_start.size() - 1;
  sums.resize(count);
  for (Eigen::Index row = 0; row < count; row++) {
    Eigen::Index s = row_start[row];
    Eigen::Index len = row_start[row + 1] - s;
    sums[row] = summand.segment(s, len).sum();
  }
}

template <typename T, typename T2>
void remove_max(const std::vector<size_t> &row_start,
                const array_t<T> &exponents, array_t<T2> &helper,
                array_t<T> &maxes) {
  // locate and remove maxes
  for (size_t row = 0; row + 1 < row_start.size(); row++) {
    Eigen::Index s = row_start[row];
    Eigen::Index len = row_start[row + 1] - s;
    if (len > 0) {
      T max = exponents.segment(s, len).maxCoeff();
      helper.segment(s, len) =
          (exponents.segment(s, len) - max).template cast<float>();
      maxes[row] = max;
    }
  }
}

template <typename T, typename result_t>
void log_sum_exp(const std::vector<size_t> &row_start,
                 const array_t<T> &exponents, result_t &result) {
  Eigen::Index count = row_start.size() - 1;
  auto &sums = eigen_memory_pool_t::get_float(count);
  auto &exp_helpers = eigen_memory_pool_t::get_second_float(exponents.size());
  result.resize(count);
  // it seems to be important to do this subtraction in float
  remove_max(row_start, exponents, exp_helpers, result);
  exp_helpers = exp_helpers.exp();
  sum(row_start, exp_helpers, sums);
  sums = sums.log();

  for (Eigen::Index i = 0; i < count; i++) {
    if (std::isfinite(result[i])) {
      result[i] += sums[i];
    }
  }
}

template <typename T1, typename T2>
array_t<real_t> componentwise_log_sum_exp(const T1 &left, const T2 &right) {
  assert(left.size() == right.size());
  Eigen::Index count = left.size();
  auto &sums = eigen_memory_pool_t::get_float(count);
  auto &exp_helpers = eigen_memory_pool_t::get_second_float(2 * count);
  array_t<real_t> result = left.max(right);
  exp_helpers.segment(0, count) = (left - result).template cast<float>();
  exp_helpers.segment(count, count) = (right - result).template cast<float>();
  exp_helpers = exp_helpers.exp();
  sums = exp_helpers.segment(0, count) + exp_helpers.segment(count, count);
  sums = sums.log();

  for (Eigen::Index i = 0; i < count; i++) {
    if (std::isfinite(result[i])) {
      result[i] += sums[i];
    }
  }
  return result;
}

template <typename T>
void sum_exp(const std::vector<size_t> &row_start, const array_t<T> &exponents,
             array_t<T> &result) {
  auto &helper = eigen_memory_pool_t::get_float(exponents.size());
  helper = exponents.template cast<float>().exp();
  sum(row_start, helper, result);
}

template <typename T> T subtract_maximum_value(array_t<T> &v) {
  T max = v.maxCoeff();
  if (max == -inf<T>()) {
    v.fill(T());
    return max;
  }
  if (max == inf<T>()) {
    for (Eigen::Index i = 0; i < v.size(); i++) {
      if (v[i] == inf<T>()) {
        v[i] = 0;
      }
      else {
        v[i] = -inf<T>();
      }
    }
    return inf<T>();
  }
  v -= max;
  return max;
}

template <typename T> T log_sum_exp(const array_t<T> &v) {
  if (v.size() == 0) { return -inf<T>(); }
  if (v.size() == 1) { return v[0]; }

  T max = v.maxCoeff();
  if (!std::isfinite(max)) {
    return max;
  }
  return std::log((v - max).exp().sum()) + max;
}

template <typename T> T log_sum_exp(const array_t<T> &v,
                                    const array_t<T> &weights) {
  if (v.size() == 0) { return -inf<T>(); }
  T max = v.maxCoeff();
  if (!std::isfinite(max)) {
    return max;
  }
  double sum = ((v - max).exp() * weights).sum();
  return std::log(sum) + max;
}

// sparse operations

inline sparse_matrix_t build_sparse_matrix(const triplet_list_t &triplet_list,
                                           size_t rows, size_t cols) {
  sparse_matrix_t M(rows, cols);
  M.setFromTriplets(triplet_list.begin(), triplet_list.end());
  M.makeCompressed();
  return M;
}

inline sparse_matrix_t build_sparse_matrix(const triplet_list_t &triplet_list,
                                           size_t size) {
  return build_sparse_matrix(triplet_list, size, size);
}

inline vector_t solve_qr(const sparse_matrix_t &M, const vector_t &b) {
  Eigen::SparseQR<Eigen::SparseMatrix<real_t>, Eigen::COLAMDOrdering<int>>
      solver;
  solver.compute(M);
  if (solver.info() != Eigen::Success) {
    throw std::out_of_range("Permutation setup failed");
  }
  vector_t x = solver.solve(b);
  if (solver.info() != Eigen::Success) {
    throw std::out_of_range("QR solver failed");
  }
  return x;
}
