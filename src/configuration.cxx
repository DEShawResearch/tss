#include <ark/ark.hpp>
#include "configuration.hxx"
#include "core/covariance_estimator.hxx"
#include "core/graph_init.hxx"
#include "core/move_types.hxx"
#include "util/util.hxx"
#include "core/communicator.hxx"
#include "core/flat_histogram.hxx"
#include "core/tss.hxx"
#include "models/gaussians.hxx"

void configure_tss(tss &sampler, Ark::ark graph_ark, bool quiet,
                   size_t num_replicas, double alpha, size_t epoch_count,
                   size_t moves_per_estimator_update, bool coordinate_invariant,
                   double visit_control_eta, double weight_epsilon,
                   double epsilon_pi) {
  if (quiet) { tss_quiet_output::enable(); }
  else { tss_quiet_output::disable(); }

  std::vector<std::shared_ptr<estimator_t>> estimator_factories;
  if (coordinate_invariant) {
    estimator_factories.push_back(std::make_shared<covariance_estimator_t>());
  } else {
    estimator_factories.push_back(std::make_shared<flat_distribution_t>());
  }

  std::shared_ptr<communicator_t> comm = std::make_shared<local_comm_t>();
  Ark::reader a(graph_ark);
  graph_t graph;
  graph.init(a);

  // create the sampler
  sampler.init(graph, num_replicas, estimator_factories, comm);
  sampler._epochs.configure(alpha, epoch_count);
  sampler._moves_per_estimator_update = moves_per_estimator_update;
  sampler._free_energies._visit_control_eta = visit_control_eta;
  sampler._free_energies._weight_epsilon = weight_epsilon;
  sampler._free_energies._log_epsilon_pi = std::log(epsilon_pi);
}

void configure_gaussians_model(tss &sampler, unsigned int dim,
                               double decorrelation, size_t seed) {
  rng_manager_t::singleton().generator.seed(seed);

  size_t num_replicas = sampler._replicas.size();
  std::vector<std::string> starting_rungs(num_replicas, "");
  std::vector<std::shared_ptr<state_t>> states(num_replicas);
  for (auto &state : states) {
    state = std::make_shared<gaussians_state>(dim, decorrelation);
  }
  sampler.populate_replicas(starting_rungs, states);
}

