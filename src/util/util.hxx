#pragma once

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <boost/container_hash/hash.hpp>

namespace desres {
namespace oats {
class access;
}
} // namespace desres

typedef double real_t;
typedef std::pair<size_t, size_t> edge_rung_t;
typedef std::pair<size_t, size_t> window_rung_t;
typedef std::pair<size_t, real_t> lambda_t;

typedef std::string schedule_key;
typedef std::string term_key;
typedef std::vector<std::pair<schedule_key, real_t>> term_schedule_t;
typedef std::map<schedule_key, double> schedule_t;

template <typename T> T quiet_nan() {
  return std::numeric_limits<T>::quiet_NaN();
}

template <typename T> T inf() { return std::numeric_limits<T>::infinity(); }

template <typename T>
void check_finite(const T &value, const std::string msg = "not_finite") {
  if (!std::isfinite(value)) {
    throw std::out_of_range(msg);
  }
}

struct schedule_hash {
  std::size_t operator()(const schedule_t &schedules) const {
    if (schedules.size() == 0) {
      return 0;
    }
    size_t hashval = 0;
    for (const auto &schedule : schedules) {
      boost::hash_combine(hashval, schedule.first);
      boost::hash_combine(hashval, schedule.second);
    }
    return hashval;
  }
};

template <typename A> struct vector_hash {
  std::size_t operator()(const std::vector<A> &vec) const {
    if (vec.size() == 0) {
      return 0;
    }
    size_t hashval = 0;
    for (const auto &element : vec) {
      boost::hash_combine(hashval, element);
    }
    return hashval;
  }
};

template <typename A, typename B> struct pair_hash {
  std::size_t operator()(const std::pair<A, B> &pair) const {
    size_t hashval = std::hash<A>()(pair.first);
    boost::hash_combine(hashval, pair.second);
    return hashval;
  }
};

template <typename A, typename B> struct pair_less {
  typedef std::pair<A, B> T;
  typedef pair_hash<A, B> hash_t;
  bool operator()(const T& lhs, const T& rhs) const {
    return hash_t()(lhs) < hash_t()(rhs);
  }
};

typedef std::map<std::pair<size_t, size_t>, real_t, pair_less<size_t, size_t>>
    pair_to_value_map_t;

class tss_quiet_output {
public:
  static bool enabled() { return enabled_; }
  static void disable() { enabled_ = false; }
  static void enable() { enabled_ = true; }

private:
  static bool enabled_;
};

inline void tss_log_printf(char const *fmt, ...) {
  if (tss_quiet_output::enabled()) {
    return;
  }

  va_list ap;
  va_start(ap, fmt);
  if (fmt)
    vfprintf(stderr, fmt, ap);
  else
    fprintf(stderr, "(null)");
  va_end(ap);
}

inline void error_printf(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);
  throw std::runtime_error("throwing");
}

// map lookup logging
template <typename container_t>
auto &logged_at(container_t &map, const typename container_t::key_type &key,
                const std::string &mapname) {
  auto result = map.find(key);
  if (result == map.end()) {
    std::ostringstream keystr;
    keystr << key;
    throw std::out_of_range("Key \"" + keystr.str() + "\" not found in " +
                            mapname);
  }
  return result->second;
}

template <typename container_t>
const auto &logged_at(const container_t &map,
                      const typename container_t::key_type &key,
                      const std::string &mapname) {
  auto result = map.find(key);
  if (result == map.end()) {
    std::ostringstream keystr;
    keystr << key;
    throw std::out_of_range("Key \"" + keystr.str() + "\" not found in " +
                            mapname);
  }
  return result->second;
}

// cleaner fill
template<typename container_t, typename value_t>
void fill(container_t& container, const value_t& value) {
  std::fill(container.begin(), container.end(), value);
}

inline void fill_with_strided_indices(std::vector<size_t> &indices,
                                      size_t count, size_t stride) {
  indices.resize(count + 1);
  for (size_t i = 0; i < count; i++) {
    indices[i] = stride * i;
  }
  indices.back() = stride * count;
}
