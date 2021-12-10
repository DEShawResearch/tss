#pragma once

#include "../util/util.hxx"
#include "../util/eigen.hxx"

struct state_t : public std::enable_shared_from_this<state_t> {
  size_t rindex;

  virtual ~state_t() {}
  virtual void init(const size_t replica_index) { rindex = replica_index; }

  virtual void update(const schedule_t& schedules,
                      const size_t /*grung*/) = 0;
  virtual array_t<real_t>
  energies(const std::vector<schedule_t>& schedules,
           const std::vector<size_t>& grungs) {
    return observables(schedules, grungs).at("energy");
  }

  virtual std::unordered_map<std::string, array_t<real_t>>
  observables(const std::vector<schedule_t> &schedules,
              const std::vector<size_t> & /*grungs*/) = 0;
};
