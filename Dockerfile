FROM ubuntu:20.04
RUN apt-get update
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y build-essential python3.7-dev libeigen3-dev git python3-pip pybind11-dev doxygen libsqlite3-dev libboost1.67-dev libboost-filesystem1.67-dev libboost-iostreams1.67-dev pandoc jupyter
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN python3.7 -m pip install numpy scipy nbsphinx scons sphinx_argparse sphinx_rtd_theme matplotlib
RUN python3.7 -m pip install pyzmq --force-reinstall
RUN mkdir /app && cd /app && git clone https://github.com/DEShawResearch/msys.git
RUN export PYTHONPATH=external:$PYTHONPATH && export PYTHONVER=37 && cd /app/msys && scons -j4 install PREFIX=/tmp/installs
RUN cd /app && git clone https://github.com/DEShawResearch/random123.git && cp -r random123/include/Random123 /usr/include/
RUN cd /app && git clone https://github.com/DEShawResearch/ark.git
RUN cd /app/ark && ./install.sh /tmp/installs
RUN cd /app && git clone https://github.com/DEShawResearch/tss.git
RUN export EIGENPATH=/usr && cd /app/tss && ./install.sh /tmp/installs
