#!/bin/bash

set -e

DNNL_VERSION=$1
DNNL_DIR=v${DNNL_VERSION}

# Check if we have clang or gcc
if [ -n "$CC" ]; then
    cc=$CC
elif which clang > /dev/null 2>&1; then
    cc=clang
elif which gcc > /dev/null 2>&1; then
    cc=gcc
else
    echo Could not find clang or gcc in '$PATH'
    exit 1
fi

# Check if we have clang++ or g++
if [ -n "$CXX" ]; then
    cxx=$CXX
elif which clang++ > /dev/null 2>&1; then
    cxx=clang++
elif which g++ > /dev/null 2>&1; then
    cxx=g++
else
    echo Could not find clang++ or g++ in '$PATH'
    exit 1
fi

# If we have clang, use libc++ and link with c++abi
CMAKE_ARGS=""
if eval ${cxx} --version | grep -q clang; then
    CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_CXX_FLAGS="-stdlib=libc++" -DCMAKE_EXE_LINKER_FLAGS="-lc++abi""
fi

( wget https://github.com/intel/mkl-dnn/archive/${DNNL_DIR}.tar.gz \
  && tar -xzf ${DNNL_DIR}.tar.gz && rm ${DNNL_DIR}.tar.gz && cd mkl-dnn-${DNNL_VERSION} \
  && mkdir -p build && cd build \
  && cmake .. ${CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Release -GNinja -DCMAKE_INSTALL_PREFIX=/tmp/dnnl_install \
  && ninja && ninja install && ldconfig )

if [ "$?" != "0" ] ; then
  exit 1
fi


