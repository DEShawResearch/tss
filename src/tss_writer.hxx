#pragma once

#include <msys/molfile/dtrplugin.hxx>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include "core/tss.hxx"


typedef desres::molfile::dtr::Key etr_key_t;

template<typename T>
struct etr_key_mapping {};

template<> struct etr_key_mapping<float> {
  enum {TYPE = etr_key_t::TYPE_FLOAT32};
};

template<> struct etr_key_mapping<double> {
  enum {TYPE = etr_key_t::TYPE_FLOAT64};
};

template<> struct etr_key_mapping<uint32_t> {
  enum {TYPE = etr_key_t::TYPE_UINT32};
};

class tss_writer
{
  typedef desres::molfile::DtrWriter DtrWriter;
  std::map<size_t, std::shared_ptr<DtrWriter>> _etrs;

public:
  enum {
    TSS_FE = 1 << 0,
    TSS_Tilts = 1 << 1,
    TSS_DetCov = 1 << 2,
    TSS_RawDetCov = 1 << 3,
    TSS_PairErrors = 1 << 4,
    TSS_RungErrors = 1 << 5,
    TSS_WindowShift = 1 << 6,
    TSS_PairFE = 1 << 7,
    TSS_UH = 1 << 8
  };

  int _output_types;

  tss_writer(int output_types = TSS_FE | TSS_Tilts | TSS_DetCov |
                                TSS_PairErrors | TSS_UH)
      : _output_types(output_types) {}

  void create_replica_etr(size_t ri) {
    mkdir(std::to_string(ri).c_str(),
          S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    std::stringstream path;
    path << ri << "/energy.etr";
    // NOTE: make_shared results in different behavior (in particular, sets
    // traj_type incorrectly)
    _etrs[ri] = std::shared_ptr<DtrWriter>(
        new DtrWriter(path.str(), DtrWriter::Type::ETR, 0, DtrWriter::CLOBBER));
    //_etr = std::make_shared<DtrWriter>(_path, DtrWriter::Type::ETR, 0,
    //   DtrWriter::APPEND);
  }

  void ensure_etrs(tss &s) {
    s.local_replica_execute([&](auto &/*replica*/, size_t ri) {
      if (_etrs.find(ri) == _etrs.end()) {
        create_replica_etr(ri);
      }
    });
  }

  void write_step(tss &s, double time) {
    ensure_etrs(s);
    s.local_replica_execute([&](auto &replica, size_t ri) {
      // let's write rung, window, and energies for window
      auto window_index = replica._last_window_index;
      const auto& energies = s._replica_rung_observables["energy"][ri];
      array_t<real_t> tss_uh;
      size_t rung = replica._rung;
      size_t last_rung = replica._last_rung;

      // Write to ETR keys
      desres::molfile::dtr::KeyMap key_map;
      key_map["TSS_rung_after"] =
          etr_key_t(&rung, 1, etr_key_t::TYPE_UINT32, false);
      key_map["TSS_rung_before"] =
          etr_key_t(&last_rung, 1, etr_key_t::TYPE_UINT32, false);
      key_map["TSS_window"] =
          etr_key_t(&window_index, 1, etr_key_t::TYPE_UINT32, false);
      if (_output_types & TSS_UH) {
        tss_uh.setConstant(s._graph.rung_count(), quiet_nan<real_t>());
        for (const auto& rung_energy : energies) {
          tss_uh[rung_energy.first] = rung_energy.second;
        }
        key_map["TSS_UH"] = etr_key_t(tss_uh.data(), tss_uh.size(),
                                      etr_key_mapping<real_t>::TYPE, false);
      }

      // If we are the first replica, compute and write unified FE information.
      // fes and weights must be defined outside of the if scope so they don't
      // go out of scope before the data is actually written in the append below
      array_t<real_t> fes;
      array_t<real_t> fs;
      array_t<real_t> weights;
      array_t<real_t> raw_weights;
      array_t<real_t> tilts;
      std::vector<uint32_t> pair_error_indices;
      array_t<real_t> pair_errors;
      array_t<real_t> rung_errors;
      array_t<real_t> pair_fes;
      if (ri == 0) {
        // get basic info
        s.update_reporting();
        if (_output_types & TSS_FE) {
          fes = s._expectations;
          key_map["TSS_FE"] = etr_key_t(fes.data(), fes.size(),
                                        etr_key_mapping<real_t>::TYPE, false);
        }
        if (_output_types & TSS_WindowShift) {
          fs = s._free_energies._f;
          key_map["TSS_WindowShift"] = etr_key_t(
              fs.data(), fs.size(), etr_key_mapping<real_t>::TYPE, false);
        }
        if (_output_types & TSS_Tilts) {
          tilts = s._free_energies.unified_tilts();
          key_map["TSS_Tilts"] = etr_key_t(
              tilts.data(), tilts.size(), etr_key_mapping<real_t>::TYPE, false);
        }
        if (_output_types & TSS_DetCov) {
          weights = s._weights;
          key_map["TSS_DetCov"] =
              etr_key_t(weights.data(), weights.size(),
                        etr_key_mapping<real_t>::TYPE, false);
        }
        if (_output_types & TSS_RawDetCov) {
          raw_weights = s._free_energies.unified_detcov();
          key_map["TSS_RawDetCov"] =
              etr_key_t(raw_weights.data(), raw_weights.size(),
                        etr_key_mapping<real_t>::TYPE, false);
        }
        if (_output_types & TSS_PairErrors) {
          // get error info
          const auto &indexed_pair_errors = s._pair_errors;
          pair_error_indices.resize(indexed_pair_errors.size() * 2);
          pair_errors.resize(indexed_pair_errors.size());
          size_t eindex = 0;
          for (const auto &pair_error : indexed_pair_errors) {
            pair_error_indices[2 * eindex] = pair_error.first.first;
            pair_error_indices[2 * eindex + 1] = pair_error.first.second;
            pair_errors[eindex] = pair_error.second;
            eindex++;
          }

          key_map["TSS_PairErrorIndices"] =
              etr_key_t(pair_error_indices.data(), pair_error_indices.size(),
                        etr_key_mapping<uint32_t>::TYPE, false);
          key_map["TSS_PairErrors"] =
              etr_key_t(pair_errors.data(), pair_errors.size(),
                        etr_key_mapping<real_t>::TYPE, false);
        }
        if (_output_types & TSS_RungErrors) {
          rung_errors = s._free_energies._rung_errors;
          key_map["TSS_RungErrors"] =
              etr_key_t(rung_errors.data(), rung_errors.size(),
                        etr_key_mapping<real_t>::TYPE, false);
        }
        if (_output_types & TSS_PairFE) {
          if (!(_output_types & TSS_PairErrors)) {
            const auto &indexed_pair_errors = s._pair_errors;
            pair_error_indices.resize(indexed_pair_errors.size() * 2);
            size_t eindex = 0;
            for (const auto &pair_error : indexed_pair_errors) {
              pair_error_indices[2 * eindex] = pair_error.first.first;
              pair_error_indices[2 * eindex + 1] = pair_error.first.second;
              eindex++;
            }
          }
          pair_fes.resize(pair_error_indices.size());
          for (int ii = 0; ii < pair_fes.size(); ii++) {
            pair_fes[ii] = s._expectations[pair_error_indices[ii]];
          }
          key_map["TSS_PairFE"] =
              etr_key_t(pair_fes.data(), pair_fes.size(),
                        etr_key_mapping<real_t>::TYPE, false);
        }
        key_map["TSS_solve_iterations"] =
            etr_key_t(&s._free_energies._last_solve_iterations, 1,
                      etr_key_mapping<uint32_t>::TYPE, false);
        key_map["TSS_solve_error"] =
            etr_key_t(&s._free_energies._last_solve_error, 1,
                      etr_key_mapping<real_t>::TYPE, false);
      }

      // Set the frame time
      _etrs[ri]->append(time, key_map);
      // I've been told this will ruin our equipment
      //_etrs[replica.index()]->sync();
    });
  }

  void stop() {
    _etrs.clear();
  }
};
