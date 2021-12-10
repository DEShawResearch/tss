#pragma once

#include <random>

struct rng_manager_t {
  std::seed_seq seed = {12, 3};
  std::mt19937 generator;

  rng_manager_t() : generator(seed) {}

  static rng_manager_t& singleton();
};

template<class T>
inline T uniform()
{
  static std::uniform_real_distribution<T> distribution(0, 1);
  return distribution(rng_manager_t::singleton().generator);
}

template<class T>
inline T gaussian(T mean = 0, T std = 1)
{
  std::normal_distribution<T> distribution(mean, std);
  return distribution(rng_manager_t::singleton().generator);
}

