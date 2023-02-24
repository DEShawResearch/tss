# Times Square Sampling
Times Square Sampling (TSS for short) is a library for performing enhanced sampling via parameter tempering.  "Times Square" is a reference to the location in Manhattan, not to the square of time.

Quick build
-----------

To build TSS, you will need the following on a Unix-family operating system:

 * A recent gcc compiler supporting C++14; we have built with gcc 8.1.0.

 * python 3.7 or greater (https://www.python.org/).

 * Scons (https://scons.org), a build tool, available through `pip install`.

 * Eigen (http://eigen.tuxfamily.org/), preferably version 3.3.7.

 * pybind11 (https://pybind11.readthedocs.io/en/stable/).

 * ark (https://github.com/DEShawResearch/msys), a configuration file library.

 * Random123 (https://github.com/DEShawResearch/random123), a counter-based random number generator library.

 * msys (https://github.com/DEShawResearch/msys), for writing and reading ETR output files.

 * boost (https://www.boost.org/), for hash combining.

 * Sphinx (https://www.sphinx-doc.org/) along with the `nbsphinx` extension (available via `pip install nbsphinx`), to build the documentation.

Make sure that scons, g++, and python are all available in your path.  Set each of EIGENPATH, PYBIND11PATH, ARKPATH, RANDOM123PATH, MSYSPATH, and BOOSTPATH, respectively, to each library's root install directory (i.e., the directory one up from the `install` or `lib` subdirectories).  Once you've done this setup, build TSS by running

    ./install.sh <install directory>

This should build the TSS library, executables, and documentation in \<install directory\>.

Other dependencies
------------------

Running TSS via the python bindings will require both numpy and scipy.  We recommended installing them via `pip install numpy scipy`.

If you wish to run the OpenMM example described in the documentation, you will also need to install OpenMM (see documentation at http://openmm.org).  This is not required to build TSS.

Installing in a container
-------------------------

You can install TSS in a container by installing Docker (https://www.docker.com) and running

    docker pull ubuntu:20.04
    docker build -t tss-context:v1 -f Dockerfile .

To see that it successfully installed and built, run

    docker run -it tss-context:v1 /tmp/installs/bin/gaussians -g /app/tss/examples/graph.cfg

You should see 157 lines of output, starting with "FREE ENERGIES".

To launch the SI notebook example, run

    docker run --workdir /app/tss -p 8888:8888 -e PYTHONPATH=/tmp/installs/lib/python -it tss-context:v1 jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

follow the console instructions to connect to the notebook in a browser, and navigate to examples/Section4_numerical_example/gaussians.ipynb.
