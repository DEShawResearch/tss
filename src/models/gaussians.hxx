#pragma once

#include "../core/state.hxx"
#include "../util/distributions.hxx"
#include "../util/util.hxx"
#include <sstream>
#include <iostream>

class gaussians_state : public state_t
{
  // internal state (current sample)
  std::vector<real_t> _value; // position

  // internal state (current generator parameters)
  unsigned int _dim; // dimensionality of the gaussians
  std::vector<real_t> _mean;
  std::vector<real_t> _deviation;
  real_t _correlation;

public:
  gaussians_state(unsigned int dim=1, real_t decorrelation=inf<real_t>()) {
    _dim = dim;
    _value = std::vector<real_t>(dim, 0);
    _mean = std::vector<real_t>(dim, 0);
    _deviation = std::vector<real_t>(dim, 1);
    _correlation = std::exp(-decorrelation);
  }

  std::vector<real_t> value() const {
    return _value;
  }

  std::vector<real_t>& value() {
    return _value;
  }

  std::pair<std::vector<real_t>, std::vector<real_t>> 
  apply_schedules(const schedule_t& schedules) {
    std::pair<std::vector<real_t>, std::vector<real_t>> 
      active = std::make_pair(_mean, _deviation);
    for (auto schedule : schedules) {
        std::istringstream name(schedule.first);
        // split_name[0] contains either "mean" or "deviation"
        // split_name[1] contains an unsigned int corresponding to the dimension
        std::vector<std::string> split_name;
        std::string segment;
        for(int i = 0; (i < 2) && (std::getline(name, segment, '_')); ++i) {
           split_name.push_back(segment);
        }
      if(split_name.size() != 2) {
        throw std::runtime_error("Tempering names must be of the form mean_i \
          or deviation_i, where i is a non-negative integer \
          (got " + name.str() + ")");
      }
      
      // identify the dimension index we are tempering
      int idx = std::stoi(split_name[1]);
      // check if mean or deviation
      if (split_name[0] == "mean") {
        active.first[idx] = schedule.second;
      }
      else if (split_name[0] == "deviation") {
        active.second[idx] = schedule.second;
      }
      else {
        throw std::runtime_error(name.str() 
            + " is not a supported temper type");
      }
    }
    return active;
  }

  void update(const schedule_t& schedules, const size_t /*grung*/) override {
    std::vector<real_t> active_mean, active_deviation;
    std::tie(active_mean, active_deviation) = apply_schedules(schedules);

    real_t new_weight = std::sqrt(1 - _correlation * _correlation);
    // harmonic oscillator in each dimension 
    for(unsigned int i = 0; i < _dim; ++i) {
      _value[i] = active_mean[i] 
                + new_weight * gaussian<real_t>(0, active_deviation[i]) 
                +_correlation * (_value[i] - active_mean[i]);
     }
    tss_log_printf("UPDATED VALUE:\n");
    for(unsigned int i = 0; i < _dim; ++i) {
      tss_log_printf("%f\n", _value[i]);  
    }
  }

  real_t energy(const schedule_t& schedules) {
    std::vector<real_t> active_mean, active_deviation;
    std::tie(active_mean, active_deviation) = apply_schedules(schedules);
    real_t val = 0;
    for(unsigned int i = 0; i < _dim; ++i) {
      val += pow((_value[i] - active_mean[i]) / (active_deviation[i]), 2);
    }
    return 0.5*val;
  }

  std::unordered_map<std::string, array_t<real_t>>
  observables(const std::vector<schedule_t> &schedules,
              const std::vector<size_t> &/*rungs*/) override {
    array_t<real_t> resulting_energies(schedules.size());
    for (size_t i = 0; i < schedules.size(); i++) {
      resulting_energies[i] = energy(schedules[i]);
    }
    // pack observables
    std::unordered_map<std::string, array_t<real_t>> observables;
    observables["energy"] = resulting_energies;
    return observables;
  }
};
