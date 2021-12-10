#pragma once

#include <set>
#include "../util/util.hxx"

namespace Ark {
class reader;
}

struct graph_t {
  std::map<std::string, std::vector<real_t>> rung_schedules;
  std::vector<std::vector<size_t>> window_rungs;
  std::vector<std::vector<size_t>> rung_to_windows;
  std::vector<std::vector<schedule_t>> window_schedules;
  // indexed by window, then overall rung, to produce rung index in window
  std::vector<std::unordered_map<size_t, size_t>> window_rung_to_ri;

  // window move groups.  per window, per dimension
  // in each window, in each dimension, there will be a vector of rung groups
  // (each itself a vector)
  // let's keep the move groups separately
  std::vector<std::vector<size_t>> rung_move_groups;
  // window move groups.  per window, per dimension, per rung
  std::vector<std::vector<std::unordered_map<size_t, size_t>>>
      window_move_groups;

  std::vector<std::vector<std::array<size_t, 3>>> invariance_adjacency;
  std::vector<real_t> rung_volumes;
  std::map<std::string, size_t> name_to_rung;
  std::vector<std::vector<size_t>> window_evaluation_rungs;
  // {{wi, wj}, [{rung, wi rung index, wj rung_index}]
  std::vector<std::pair<std::pair<size_t, size_t>,
                        std::vector<std::tuple<size_t, size_t, size_t>>>>
      intersecting_windows;

  void init(Ark::reader a);

  size_t rung_count() const { return rung_volumes.size(); }

  size_t rung_dimension(const size_t rung) const {
    return invariance_adjacency[rung].size();
  }

  size_t rung_from_rung_name(const std::string &rungname) const {
    if (name_to_rung.find(rungname) == name_to_rung.end()) {
      throw std::invalid_argument("rung name \"" + rungname + "\" not found");
    }
    return name_to_rung.at(rungname);
  }

  schedule_t rung_to_schedules(const size_t rung) const {
    schedule_t schedule;
    for (const auto &entry : rung_schedules) {
      schedule[entry.first] = entry.second[rung];
    }
    return schedule;
  }

  size_t swap_window(size_t window_index, size_t rung) const {
    const auto &rung_windows = rung_to_windows[rung];
    for (size_t i = 0; i < rung_windows.size(); i++) {
      if (rung_windows[i] == window_index) {
        return rung_windows[(i + 1) % rung_windows.size()];
      }
    }
    throw std::out_of_range("Attempting to swap from a window that does not "
                            "exist at this rung");
  }

  std::set<size_t> window_rung_set(size_t window_index) const {
    std::set<size_t> rungs;

    // gather the set of rungs to evaluate
    for (auto r : window_rungs[window_index]) {
      rungs.insert(r);
      for (const auto &dim_neighbors : invariance_adjacency[r]) {
        for (size_t i = 0; i < 2; i++) {
          rungs.insert(dim_neighbors[i]);
        }
      }
    }
    return rungs;
  }
};
