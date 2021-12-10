#include <fstream>
#include <unistd.h>
#include <ark/ark.hpp>
#include "core/move_types.hxx"
#include "core/tss.hxx"
#include "configuration.hxx"
#include "tss_writer.hxx"

void show_usage(std::string name) {
  std::cerr
      << "Usage: " << name << " <option(s)>\n"
      << "Options:\n"
      << "\t-h\t\tShow this help message\n"
      << "\t-l\t\tNumber of steps to run (default 1000)\n"
      << "\t-q\t\tWhether to run with quiet output (default 1)\n"
      << "\t-w\t\tWhether to write ETR (default 1)\n"
      << "\t-i\t\tStride the ETR output (default stride = 1)\n"
      << "\t-r\t\tNumber of replicas (default 1)\n"
      << "\t-g\t\tGraph file name (default 'graph.cfg')\n"
      << "\t-f\t\tWhether to use the flat target rung distribution "
         "(default 1)\n"
      << "\t-a\t\tEpoch alpha (default 0.19)\n"
      << "\t-e\t\tNumber of active epochs (default 16)\n"
      << "\t-p\t\tMoves per estimator update (default 1)\n"
      << "\t-v\t\tVisit control eta (default 2)\n"
      << "\t-s\t\tRandom seed\n"
      << "\t-d\t\tDecorrelation at each step (default inf)\n"
      << "\t-n\t\tDimension of the gaussians (default 1)\n"
      << "\t-u\t\tInclude output (default 1)\n"
      << "\t-t\t\tRegularization weight epsilon (default 0.01)\n"
      << "\t-x\t\tRegularization epsilon_pi (default 1e-3)\n"
      << "\t-y\t\tWrite rung error to the ETR (default 0)\n"
      << "\t-k\t\tCompute errors (default 1)\n"
      << std::endl;
}

int main(int argc, char** argv) {
  unsigned int last_time = 1000;
  int quiet = 1;
  int write = 1;
  int interval = 1;
  unsigned int num_replicas = 1;
  std::string graph_file("graph.cfg");
  double alpha = 0.19;
  int epoch_count = 16;
  int moves_per_estimator_update = 1;
  double decorrelation = std::numeric_limits<double>::infinity();
  bool coordinate_invariant = false;
  double visit_control_eta = 2;
  unsigned int dimension = 1;
  int seed = 10;
  bool compute_errors = 1;
  bool include_output = 1;
  double regularization_epsilon = 0.01;
  double epsilon_pi = 1e-3;
  int output_types = tss_writer::TSS_FE | tss_writer::TSS_Tilts |
                     tss_writer::TSS_DetCov | tss_writer::TSS_PairErrors |
                     tss_writer::TSS_UH;

  int opt;
  while ((opt = getopt(argc, argv, "l:q:w:i:r:g:a:d:p:f:v:s:h:e:n:k:t:u:x:y:")) !=
         -1) {
    switch (opt) {
    case 'l':
      last_time = std::atoi(optarg);
      break;
    case 'q':
      quiet = std::atoi(optarg);
      break;
    case 'w':
      write = std::atoi(optarg);
      break;
    case 'i':
      interval = std::atoi(optarg);
      break;
    case 'r':
      num_replicas = std::atoi(optarg);
      break;
    case 'g':
      graph_file = std::string(optarg);
      break;
    case 'a':
      alpha = std::atof(optarg);
      break;
    case 'e':
      epoch_count = std::atof(optarg);
      break;
    case 'd':
      decorrelation = std::atof(optarg);
      break;
    case 'p':
      moves_per_estimator_update = std::atoi(optarg);
      break;
    case 'f':
      coordinate_invariant = std::atoi(optarg) == 0;
      break;
    case 'v':
      visit_control_eta = std::atof(optarg);
      break;
    case 's':
      seed = std::atoi(optarg);
      break;
    case 'n':
      dimension = std::atoi(optarg);
      break;
    case 'k':
      compute_errors = std::atoi(optarg);
      break;
    case 't':
      regularization_epsilon = std::atof(optarg);
      break;
    case 'x':
      epsilon_pi = std::atof(optarg);
      break;
    case 'u':
      include_output = std::atoi(optarg);
      break;
    case 'y':
      if (std::atoi(optarg)) {
        output_types |= tss_writer::TSS_RungErrors;
      }
      break;
    case 'h':
    default:
      show_usage(argv[0]);
      return 0;
    }
  }

  Ark::ark graph_ark;
  Ark::parser parser;
  parser.parse_file(graph_ark, graph_file);
  tss sampler;
  using namespace move_types;
  std::unique_ptr<move_base_t> mover(new independent_move_t);
  configure_tss(sampler, graph_ark, quiet, num_replicas, alpha, epoch_count,
                moves_per_estimator_update, coordinate_invariant,
                visit_control_eta, regularization_epsilon, epsilon_pi);
  configure_gaussians_model(sampler, dimension, decorrelation, seed);
  sampler._compute_errors = compute_errors;
  tss_writer writer;

  // main TSS loop
  for (unsigned int i = 0; i < last_time; i++) {
    sampler.step(mover.get());
    if (write && i % interval == 0) {
      writer.write_step(sampler, i);
    }
  }
  sampler.stop(!include_output);
  return 0;
}
