#!/bin/bash

set -eux

ONEDNN_VERSION=$1
ONEDNN_TAR_NAME="v${ONEDNN_VERSION}"
ONEDNN_DIR="oneDNN-${ONEDNN_VERSION}"

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

# Don't bother with building examples and tests for installing the library 
CMAKE_ARGS="-GNinja -DCMAKE_BUILD_TYPE=Release -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF"
 
# If we have clang, use libc++ and link with c++abi
if eval ${cxx} --version | grep -q clang; then
    CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_CXX_FLAGS="-stdlib=libc++" -DCMAKE_EXE_LINKER_FLAGS="-lc++abi""
fi

( wget https://github.com/intel/mkl-dnn/archive/${ONEDNN_TAR_NAME}.tar.gz && mkdir -p /tmp \
  && tar -xzf ${ONEDNN_TAR_NAME}.tar.gz -C /tmp && rm ${ONEDNN_TAR_NAME}.tar.gz \
  && mkdir -p /tmp/${ONEDNN_DIR}/build && cd /tmp/${ONEDNN_DIR}/build \
  && cmake .. ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=/tmp/onednn_install \
  && ninja && ninja install && ldconfig )

if [ "$?" != "0" ] ; then
  exit 1
fi


