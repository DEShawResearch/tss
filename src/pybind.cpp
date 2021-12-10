#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <ark/ark.hpp>

#include "core/covariance_estimator.hxx"
#include "core/communicator.hxx"
#include "core/move_types.hxx"
#include "core/flat_histogram.hxx"
#include "core/tss.hxx"

#include "configuration.hxx"
#include "tss_writer.hxx"

namespace py = pybind11;


struct DummyState : public state_t {
  friend class desres::oats::access;
  template <typename A> void cerealize(A &a) {
    a[static_cast<state_t &>(*this)];
    a(m_energies);
  }
  virtual void update(const schedule_t &, const size_t) override {}

  virtual std::unordered_map<std::string, array_t<real_t>>
  observables(const std::vector<schedule_t> & /*schedules*/,
              const std::vector<size_t> &rungs) override {
    array_t<real_t> energies(rungs.size());
    for (Eigen::Index ri = 0; ri < energies.size(); ri++) {
      energies[ri] = m_energies[rungs[ri]];
    }
    std::unordered_map<std::string, array_t<real_t>> observables;
    observables["energy"] = energies;
    return observables;
  }
  array_t<real_t> m_energies;
};


class TimesSquareSampler {
private:
  tss m_tss;
  move_types::independent_move_t m_move_ind;
  tss_writer writer;

public:
  TimesSquareSampler() {}

  TimesSquareSampler(Ark::ark graph_ark, bool quiet = true,
                     size_t num_replicas = 1, double alpha = 0.19,
                     size_t epoch_count = 16,
                     size_t moves_per_estimator_update = 1,
                     bool coordinate_invariant = true,
                     double visit_control_eta = 2,
                     double weight_epsilon = 0.01,
                     double epsilon_pi = 1e-3) {
    configure_tss(m_tss, graph_ark, quiet, num_replicas, alpha, epoch_count,
                  moves_per_estimator_update, coordinate_invariant,
                  visit_control_eta, weight_epsilon, epsilon_pi);
  }

  void initGaussiansModel(
      unsigned int dim = 1,
      double decorrelation = std::numeric_limits<double>::infinity(),
      size_t seed = 10) {
    configure_gaussians_model(m_tss, dim, decorrelation, seed);
  }

  void initDummyModel(size_t seed = 10) {
    m_move_ind._rng.seed(seed);

    unsigned int num_replicas = m_tss._replicas.size();
    std::vector<std::shared_ptr<state_t>> states;
    for (size_t i = 0; i < num_replicas; i++) {
      states.push_back(std::make_shared<DummyState>());
    }
    std::vector<std::string> starting_rungs(num_replicas);
    std::fill(starting_rungs.begin(), starting_rungs.end(), "");
    m_tss.populate_replicas(starting_rungs, states);
  }

  void setEnergies(int replica, std::vector<real_t> energies) {
    std::dynamic_pointer_cast<DummyState>(m_tss._replicas[replica]._state)
        ->m_energies = array_t<real_t>::Map(energies.data(), energies.size());
  }

  void setMove(int replica, size_t rung, bool able_to_move) {
    m_tss._replicas[replica]._unable_to_move = !able_to_move;
    m_tss._replicas[replica]._last_rung = m_tss._replicas[replica]._rung;
    m_tss._replicas[replica]._rung = rung;
  }

  void step(size_t num_steps = 1, bool write = false,
            size_t write_interval = 1) {
    for (unsigned i = 0; i < num_steps; i++) {
      m_tss.step(&m_move_ind);
      if (write && m_tss.time() % write_interval == 0) {
        writer.write_step(m_tss, m_tss.time());
      }
    }
  }

  void stop() {
    writer.stop();
  }

  int getReplicaWindow(int replica) {
    return m_tss._replicas[replica]._window_index;
  }

  int getReplicaRung(int replica) { return m_tss._replicas[replica]._rung; }

  void setReplicaRung(int replica, int rung) {
    m_tss._replicas[replica].set_window_rung(
        m_tss._graph, m_tss._graph.rung_to_windows[rung][0], rung);
  }

  std::vector<real_t> getFreeEnergies() const {
    return stdvector_from_array(m_tss._free_energies._expectations);
  }

  std::vector<real_t> getRungWeights() const {
    return stdvector_from_array(m_tss._free_energies._weights);
  }

  std::pair<std::vector<int>, std::vector<real_t>> getPairErrors() const {
    const auto &indexed_pair_errors = m_tss._free_energies._pair_errors;
    std::vector<int> pair_error_indices;
    std::vector<real_t> pair_errors;
    pair_error_indices.resize(indexed_pair_errors.size() * 2);
    pair_errors.resize(indexed_pair_errors.size());
    size_t eindex = 0;
    for (const auto &pair_error : indexed_pair_errors) {
      pair_error_indices[2 * eindex] = pair_error.first.first;
      pair_error_indices[2 * eindex + 1] = pair_error.first.second;
      pair_errors[eindex] = pair_error.second;
      eindex++;
    }
    return {pair_error_indices, pair_errors};
  }

  void updateReporting() {
    m_tss.update_reporting();
  }

  std::vector<int> getEvaluationRungs(int window_index) const {
    auto rung_set = m_tss._graph.window_rung_set(window_index);
    std::vector<int> rungs(rung_set.begin(), rung_set.end());
    return rungs;
  }

  int getNumRungs() const { return m_tss._graph.rung_count(); }
  auto getWindowRungs() const { return m_tss._graph.window_rungs; }
};

PYBIND11_MODULE(_tss, m) {
  m.doc() = "Python bindings for a subset of Times Square Sampling";

  py::class_<TimesSquareSampler>(m, "Sampler")
      .def(py::init<Ark::ark, bool, size_t, double, size_t, size_t, bool,
                    double, double, double>(),
           py::arg("graph_ark"), py::arg("quiet") = true,
           py::arg("num_replicas") = 1, py::arg("alpha") = 0.19,
           py::arg("epoch_count") = 16,
           py::arg("moves_per_estimator_update") = 1,
           py::arg("coordinate_invariant") = true,
           py::arg("visit_control_eta") = 2, py::arg("weight_epsilon") = 0.01,
           py::arg("epsilon_pi") = 1e-3,
           R"DOC(
Initialize the TSS sampler.

Args:
    graph_ark (Ark): TSS graph ark object.
    quiet (bool): Whether to suppress TSS debugging output (default True).
    num_replicas (int): The number of active replica (default 1).
    alpha (float): The fraction of history to keep (default 0.19).
    epoch_count (int): The number of epochs of history to keep (default 16).
    moves_per_estimator_update (int): The number of lambda moves to make for
        each estimator update (default 1).
    coordinate_invariant (bool): Whether to use coordinate invariant weights or
        the flat distribution (default True).
    visit_control_eta (float): The eta to use online for visit control
        (default 2).
    weight_epsilon (float): If the max rung DetCov is D, any rung DetCov that
        falls below weight_epsilon * D is clamped to it (default 0.01).
    epsilon_pi (float): Clamps the minimum (active rung weight / rung volume)
        in any window to be at least epsilon_pi * the maximum (default 1e-3).
)DOC")

      .def(py::init(), R"DOC(
Not intended to be called directly.
)DOC")

      .def("initDummyModel", &TimesSquareSampler::initDummyModel,
           py::arg("seed") = 10,
           R"DOC(
Initialize the per-replica models to a dummy model.  The dummy model does not
compute energies; use set_energies to set the energies before every TSS step.

Args:
    seed (int): The seed for the random number generator for the TSS move
        (default 10).
)DOC")

      .def("initGaussiansModel", &TimesSquareSampler::initGaussiansModel,
           py::arg("dim") = 1,
           py::arg("decorrelation") = std::numeric_limits<double>::infinity(),
           py::arg("seed") = 10,
           R"DOC(
Initialize the per-replica models to the Gaussians toy model.

Args:
    dim (int): number of Gaussians (default 1) with means and standard
        deviations controlled by "mean_N" and "deviation_N" in the graph
        where N is the 0-based index of the target Gaussian.
    decorrelation (float): How decorrelated each sample drawn from a rung is
        with the prior sample value (default inf).
    seed (int): The seed for the random number generator for the TSS move
        (default 10).
)DOC")

      .def("step", &TimesSquareSampler::step, py::arg("num_steps") = 1,
           py::arg("write") = false, py::arg("write_interval") = 1, R"DOC(
Take some number of full TSS steps.

Args:
    num_steps (int): The number of TSS steps to take (default 1).
    write (bool): Whether to write estimator output to ETR files (default 
        false).
    write_interval (int): If writing ETRs, do so every write_interval steps
        (default 1).
)DOC")

      .def("stop", &TimesSquareSampler::stop, R"DOC(
Stop and flush the ETR writer.  Necessary for loading the ETR.
)DOC")

      .def("setEnergies", &TimesSquareSampler::setEnergies, py::arg("replica"),
           py::arg("energies"), R"DOC(
Set the graph-wide energies to be used in the next TSS step.

Args:
    replica (int): Which replica's energy is being set.
    energies (numpy array): An array of per-rung energies.  Must have the same
        number of entries as rungs in the full graph, but only the energies from
        the set of evaluation rungs for the current window will be read.
)DOC")

      .def("setMove", &TimesSquareSampler::setMove, py::arg("replica"),
           py::arg("rung"), py::arg("able_to_move"), R"DOC(
Report the move that was actually performed externally.

Args:
    replica (int): Which replica's rung is being set.
    rung (int): Which rung index to set it to.
    able_to_move (bool): If the move did or did not actually occur (e.g. due to
        an uninitialized window).
)DOC")

      .def("getNumRungs", &TimesSquareSampler::getNumRungs, R"DOC(
Return the number of rungs in the full graph.

Returns:
    int: The number of rungs in the full graph.
)DOC")

      .def("getReplicaWindow", &TimesSquareSampler::getReplicaWindow,
           py::arg("replica"), R"DOC(
Get the current window index for a replica.

Args:
    replica (int): The index of a replica.

Returns:
    int: The window index for the selected replica.
)DOC")

      .def("getReplicaRung", &TimesSquareSampler::getReplicaRung,
           py::arg("replica"), R"DOC(
Get the current rung index for a replica.

Args:
    replica (int): The index of a replica.

Returns:
    int: The rung index for the selected replica.
)DOC")

      .def("setReplicaRung", &TimesSquareSampler::setReplicaRung,
           py::arg("replica"), py::arg("rung"), R"DOC(
Set the rung index for a replica.

Args:
    replica (int): The index of a replica.
    rung (int): The index of a rung.
)DOC")

      .def("getFreeEnergies", &TimesSquareSampler::getFreeEnergies, R"DOC(
Get the current estimated free energies (must call updateReporting first).

Returns:
    list (float): The current estimated free energies for all rungs.
)DOC")

      .def("getEvaluationRungs", &TimesSquareSampler::getEvaluationRungs,
           py::arg("window_index"),
           R"DOC(
Get the set of rungs whose energies must be evaluated to perform a TSS update
for a given window.

Args:
    window_index (int): The window index.

Returns:
    list (int): A list of rung indices.
)DOC")

      .def("getRungWeights", &TimesSquareSampler::getRungWeights, R"DOC(
Get the current target rung visit frequencies.

Returns:
    list (float): List holding the current target rung visit frequencies,
        indexed by rung
)DOC")

      .def("getPairErrors", &TimesSquareSampler::getPairErrors, R"DOC(
Get the set of rung index pairs whose relative free energy errors are being
estimated, along with the current values of those estimates.

Returns:
    tuple: a list of (npairs * 2) ints (giving the sets of rung pairs in groups
        of two) and a list of (npairs) floats giving the current estimates of
        the relative free energy error between each pair.  By default the pair
        of rungs (0, nrungs - 1) will be included.
)DOC")

      .def("updateReporting", &TimesSquareSampler::updateReporting, R"DOC(
Have TSS update the output free energy estimates.  Should be called before
calling getFreeEnergies.
)DOC");
}
