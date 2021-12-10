#pragma once
#include "graph.hxx"

#include <ark/ark.hpp>
#include <ark/arkTo.hpp>

inline Ark::vector_t to_ark_vector(const Ark::reader& a) {
  return a;
}

// this is in a separate file so that the library does not have to commit to
// an Ark version
void graph_t::init(Ark::reader a) {
  // load edges and nodes.  This is for rung naming.
  std::map<size_t, std::string> rung_to_node;
  auto edges_ark = to_ark_vector(a.get("graph.edges"));
  size_t edge_count = edges_ark.size();
  std::vector<std::vector<std::string>> edge_to_nodes(edge_count);
  std::vector<std::string> edge_names(edge_count);
  for (size_t edge_id = 0; edge_id < edge_count; edge_id++) {
    auto edge_ark = edges_ark[edge_id];
    auto node_set = edge_ark.get("node_set");
    std::vector<std::string> edge_nodes;
    std::function<void(const Ark::ark &)> recurse_nodes =
        [&](const Ark::ark &node_ark) {
          if (node_ark.kind() == Ark::Vector) {
            Ark::vector_t vec = node_ark.vector();
            for (size_t i = 0; i < vec.size(); i++) {
              recurse_nodes(vec[i]);
            }
          } else {
            std::string node_name = node_ark.atom();
            edge_nodes.push_back(node_name);
          }
        };
    recurse_nodes(*node_set);
    std::string edge_name = edge_nodes[0];
    for (size_t node_id = 1; node_id < edge_nodes.size(); node_id++) {
      edge_name += ":" + edge_nodes[node_id];
    }
    edge_names[edge_id] = edge_name;
  }

  std::map<std::string, std::vector<size_t>> node_to_rungs;
  auto nodes_ark = to_ark_vector(a.get("graph.nodes"));
  for (auto node_ark : nodes_ark) {
    std::string name = Ark::arkTo<std::string>(*node_ark.get("name"));
    auto rungs = node_ark.get("rung_set")->vector();
    // arbitrarily pick one of the nodes at this rung to correspond to it
    name_to_rung[name] = Ark::arkTo<size_t>(rungs[0]);
  }

  // load the rungs
  auto rungs_ark = to_ark_vector(a.get("graph.rungs"));
  size_t rung_count = rungs_ark.size();

  invariance_adjacency.resize(rung_count);
  rung_volumes.resize(rung_count);
  std::vector<std::vector<size_t>> rung_locations(rung_count);
  for (size_t rung = 0; rung < rung_count; rung++) {
    auto rung_ark = &rungs_ark[rung];
    const auto &parameter_table = rung_ark->get("parameters")->table();
    for (const auto &parameter : parameter_table) {
      auto &parameters = rung_schedules[parameter.first];
      parameters.resize(rung_count);
      parameters[rung] = Ark::arkTo<real_t>(parameter.second);
    }

    rung_volumes[rung] = Ark::arkTo<real_t>(*rung_ark, "volume");

    const auto &invariance_neighbors =
        rung_ark->get("neighbors_dimensional")->vector();
    size_t dim = invariance_neighbors.size();
    auto &rung_invariance = invariance_adjacency[rung];
    rung_invariance.resize(dim);
    for (size_t d = 0; d < dim; d++) {
      const Ark::vector_t &dim_invariance = invariance_neighbors[d].vector();
      size_t real_neighbor_count = 0;
      for (size_t i = 0; i < 2; i++) {
        rung_invariance[d][i] = Ark::arkTo<size_t>(dim_invariance[i]);
        real_neighbor_count += rung_invariance[d][i] != rung;
      }
      rung_invariance[d][2] = real_neighbor_count;
    }

    // set up rung name
    size_t edge_id = Ark::arkTo<size_t>(*rung_ark, "edge_id");
    auto &location = rung_locations[rung];
    Ark::arkToIter<size_t>(*rung_ark, "location", std::back_inserter(location));
    std::string rung_name = edge_names[edge_id];
    for (auto index : location) {
      rung_name += ":" + std::to_string(index);
    }
    name_to_rung[rung_name] = rung;
  }

  // load the windows
  Ark::vector_t windows_ark = to_ark_vector(a.get("graph.windows"));
  size_t window_count = windows_ark.size();

  window_rungs.resize(window_count);
  rung_to_windows.resize(rung_count);
  window_move_groups.resize(window_count);
  for (size_t window = 0; window < window_count; window++) {
    size_t dimension = 0;
    std::function<void(const Ark::ark &, int)> recurse_rungs =
        [&](const Ark::ark &rung_ark, int depth) {
          if (rung_ark.kind() == Ark::Vector) {
            Ark::vector_t vec = rung_ark.vector();
            for (size_t i = 0; i < vec.size(); i++) {
              recurse_rungs(vec[i], depth + 1);
            }
          } else {
            auto rung = Ark::arkTo<int>(rung_ark);
            window_rungs[window].push_back(rung);
            rung_to_windows[rung].push_back(window);
            dimension = std::max(dimension, (size_t)depth);
          }
        };
    auto top_rung_set = windows_ark[window].get("rung_set");
    recurse_rungs(*top_rung_set, -1);

    // build window move groups if all the rungs are on one edge
    if (top_rung_set->vector().size() == 1) {
      // single-edge window
      window_move_groups[window].resize(dimension);
      for (size_t dim = 0; dim < dimension; dim++) {
        //
        auto &dim_move_groups = window_move_groups[window][dim];

        // location in all other dimensions -> {location in this dim, rung}
        std::unordered_map<std::vector<size_t>,
                           std::vector<std::pair<size_t, size_t>>,
                           vector_hash<size_t>>
            dim_group_map;
        for (auto rung : window_rungs[window]) {
          auto location = rung_locations[rung];
          // remove dimension dim
          size_t dloc = location[dim];
          location.erase(location.begin() + dim);
          // add to map
          dim_group_map[location].emplace_back(dloc, rung);
        }
        // sort each group and store/register
        for (auto &group : dim_group_map) {
          std::sort(group.second.begin(), group.second.end(),
                    [](auto a, auto b) { return a.first < b.first; });
          size_t group_index = rung_move_groups.size();
          std::vector<size_t> group_rungs;
          group_rungs.reserve(group.second.size());
          for (auto loc_rung : group.second) {
            group_rungs.push_back(loc_rung.second);
            dim_move_groups[loc_rung.second] = group_index;
          }
          rung_move_groups.emplace_back(group_rungs);
        }
      }
    }
  }

  // precompute the rung schedules
  std::vector<schedule_t> rung_schedule_vector(rung_count);
  for (size_t rung = 0; rung < rung_count; rung++) {
    rung_schedule_vector[rung] = rung_to_schedules(rung);
  }

  // precompute the window rung sets
  window_evaluation_rungs.resize(window_count);
  window_schedules.resize(window_count);
  for (size_t wi = 0; wi < window_count; wi++) {
    auto rung_set = window_rung_set(wi);
    auto &werungs = window_evaluation_rungs[wi];
    auto &wschedules = window_schedules[wi];
    werungs = std::vector<size_t>(rung_set.begin(), rung_set.end());
    wschedules.resize(werungs.size());
    for (size_t ri = 0; ri < werungs.size(); ri++) {
      wschedules[ri] = rung_schedule_vector[werungs[ri]];
    }
  }

  // set up map from window rung to window rung index
  window_rung_to_ri.resize(window_count);
  for (size_t wi = 0; wi < window_count; wi++) {
    for (size_t ri = 0; ri < window_rungs[wi].size(); ri++) {
      window_rung_to_ri[wi][window_rungs[wi][ri]] = ri;
    }
  }
  // precompute the intersecting window sets
  std::unordered_map<std::pair<size_t, size_t>,
                     std::vector<std::tuple<size_t, size_t, size_t>>,
                     pair_hash<size_t, size_t>>
      intersection_map;
  for (size_t rung = 0; rung < rung_count; rung++) {
    // find all windows at this rung and add each other to the other's overlap
    const auto &ws = rung_to_windows[rung];
    for (size_t i = 0; i < ws.size(); i++) {
      auto wi = ws[i];
      for (size_t j = i + 1; j < ws.size(); j++) {
        auto wj = ws[j];
        intersection_map[{wi, wj}].push_back(
            {rung, window_rung_to_ri[wi][rung], window_rung_to_ri[wj][rung]});
      }
    }
  }
  for (const auto &intersection : intersection_map) {
    intersecting_windows.push_back({intersection.first, intersection.second});
  }
}
