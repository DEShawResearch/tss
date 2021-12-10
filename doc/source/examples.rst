Examples of how to run TSS
==========================

Running TSS via ipython
-----------------------

See the notebook below (and at ``examples/paper_appendix_numerical_examples/gaussians.ipynb``) for an example of running TSS via the python interface and a simple model and plotting the results.

.. toctree::

   gaussians.ipynb

Running the gaussians example
-----------------------------

The ``gaussians`` program runs TSS to estimate free energies for a graph where every rung corresponds to drawing samples from a gaussian distribution parameterized by a (mean, standard deviation) pair.  Running it requires first building a graph file.  If we wish to create a graph with 16 gaussians, spaced a distance of 1 apart, with standard deviation 1, and with three overall windows, we would first create a graph spec file "spec.cfg"::

    integrator {
        times_square {
            edges = [{
            nodes = ["cold" "hot"]
            number_of_rungs = "16"
            window_size = "16"
            schedule = [{
                bounds = ["1" "16"]
                group_name = "mean"
                dimension = "0"
            } {
                bounds = ["1" "1"]
                group_name = "deviation"
                dimension = "0"
            }]
        }
    }

and then convert it into a fully explicit graph file by calling

.. code-block:: console

    build_graph.py spec.cfg -o graph.cfg

You can then run the ``gaussians`` program via

.. code-block:: console

    gaussians -g graph.cfg -l 10000

which will generate 10000 frames of output in the ``0/energy.etr`` directory (run ``gaussians -h`` to see what other options are available).  See "Reading TSS output" below for details of the format of the output.  To plot the evolution of the delta free energy between the first and last rungs, run

.. code-block:: console

    plot_fe.py 0/energy.etr

Running TSS with OpenMM
-----------------------

``tests/tss-openmm-amber.py`` provides an example of how to use TSS with a molecular dynamics package, in this case OpenMM.  It is set up to take an Amber system as input, but should be easy to adapt to other systems that OpenMM supports.  A simple example that we've tried is to take the DHFR system available in the SI of the OpenMM 7 paper (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005659#sec023).  Find the files ``dhfr.pbc.parm7`` and ``dhfr.pbc.rst7`` in ``SI/Energy Comparisons/Amber``.  You can then run the example via

.. code-block:: console

    tests/tss-openmm-amber.py --prm-file dhfr.pbc.parm7 --crd-file dhfr.pbc.rst7

