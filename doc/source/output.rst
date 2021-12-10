Reading TSS output
------------------

The ``gaussians`` example writes output in ``msys`` ``Molfile`` ETR (similar to DTR) trajectory files.  See ``bin/plot_fe.py`` for an example of how to read these files and use the estimated free energies.  See the Python Reference section of the ``msys`` documentation for more general information on how to access ``Molfile`` output.  The fields that TSS writes into each frame of the ETR are

    TSS_FE (nrungs): per-rung free energy estimate
    TSS_UH (nrungs): per-rung energy evaluations for the current frame (only rungs in the evaluation set for the current window will be non-NaN)
    TSS_rung_before: rung index at the beginning of the TSS step
    TSS_rung_after: rung index at the end of the TSS step
    TSS_window: index of the window used to do the move between TSS_rung_before and TSS_rung_after
    TSS_DetCov (nrungs): per-rung estimate of the target fraction of the simulation that should be spent at that rung
    TSS_PairErrorIndices (2 * num_error_pairs): pairs of indices to estimate delta FE errors between (in the format [pair1_a, pair1_b, pair2_a, pair2_b, ...])
    TSS_PairErrors (num_error_pairs): current estimates of delta FE errors between the pairs specified in TSS_PairErrorIndices
    TSS_Tilts (nrungs): per-rung estimate of the average visit control tilts ``o``

