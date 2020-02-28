#!/bin/bash

set -e

# Find our C compiler (clang or gcc)
if [ -n "$CC" ]; then
    cc=$CC
elif which clang > /dev/null 2>&1; then
    cc=clang
elif which gcc > /dev/null 2>&1; then
    cc=gcc
else
    echo could not find clang or gcc in '$PATH'
    exit 1
fi

# Find our C++ compiler (clang++ or g++)
if [ -n "$CXX" ]; then
    cxx=$CXX
elif which clang++ > /dev/null 2>&1; then
    cxx=clang++
elif which g++ > /dev/null 2>&1; then
    cxx=g++
else
    echo could not find clang++ or g++ in '$PATH'
    exit 1
fi

CMAKE_FLAGS=""

# If we have clang, we need have to use libc++ and libc++abi
if eval ${cxx} --version | grep -q clang; then
    CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_CXX_FLAGS=\"-stdlib=libc++\" -DCMAKE_EXE_LINKER_FLAGS=\"-lc++abi\""
fi


git clone https://gitlab.lrz.de/IP/elsa.git /tmp/elsa/

mkdir -p /tmp/elsa/build && cd /tmp/elsa/build/

echo "cmake .. -GNinja ${CMAKE_FLAGS}"
cmake .. -GNinja ${CMAKE_FLAGS}
ninja docs
