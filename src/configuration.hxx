#pragma once

#include <limits>

class tss;
namespace Ark {
  class ark;
}

void configure_tss(tss &sampler, Ark::ark graph_ark, bool quiet = 1,
                   size_t num_replicas = 1, double alpha = 0.19,
                   size_t epoch_count = 16,
                   size_t moves_per_estimator_update = 1,
                   bool coordinate_invariant = true,
                   double visit_control_eta = 2,
                   double weight_epsilon = 0.01,
                   double epsilon_pi = 0.001);

void configure_gaussians_model(
    tss &sampler,
    unsigned int dim = 1,
    double decorrelation = std::numeric_limits<double>::infinity(),
    size_t seed = 10);
