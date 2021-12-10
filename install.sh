#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 TARGET_DIRECTORY" >&2
    exit 1
fi

PREFIX=$1

# set up build variables
export PYTHONPATH=external:$PYTHONPATH

if [[ ! -z $EIGENPATH ]]; then
    export DESRES_MODULE_CPPFLAGS="-I$EIGENPATH/include/eigen3":$DESRES_MODULE_CPPFLAGS
fi

if [[ ! -z "${PYBIND11PATH}" ]]; then
    export DESRES_MODULE_CPPFLAGS="-I$PYBIND11PATH/include":$DESRES_MODULE_CPPFLAGS    
fi

if [[ ! -z "${ARKPATH}" ]]; then
    export DESRES_MODULE_CPPFLAGS="-I$ARKPATH/include":$DESRES_MODULE_CPPFLAGS
    export DESRES_MODULE_LDFLAGS="-L$ARKPATH/lib":$DESRES_MODULE_LDFLAGS
    export DESRES_MODULE_LDFLAGS=-Wl,-rpath=$ARKPATH/lib:$DESRES_MODULE_LDFLAGS
    export PYTHONPATH=$ARKPATH/lib/python:$PYTHONPATH
fi

if [[ ! -z "${RANDOM123PATH}" ]]; then
    export DESRES_MODULE_CPPFLAGS="-I$RANDOM123PATH/include":$DESRES_MODULE_CPPFLAGS
fi

if [[ ! -z "${MSYSPATH}" ]]; then
    export DESRES_MODULE_CPPFLAGS="-I$MSYSPATH/include":$DESRES_MODULE_CPPFLAGS
    export DESRES_MODULE_LDFLAGS="-L$MSYSPATH/lib":$DESRES_MODULE_LDFLAGS
    export DESRES_MODULE_LDFLAGS=-Wl,-rpath=$MSYSPATH/lib:$DESRES_MODULE_LDFLAGS
fi

if [[ ! -z "${BOOSTPATH}" ]]; then
    export DESRES_MODULE_CPPFLAGS="-I$BOOSTPATH/include":$DESRES_MODULE_CPPFLAGS
fi

# invoke build
. version.sh
scons -j4 install PREFIX=$PREFIX/ PYTHONVER=37 VERSION=$VERSION

# build documentation
mkdir -p $PREFIX/doc
(cd doc && LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH PREFIX=$PREFIX make BUILDDIR=$PREFIX/doc clean html)

