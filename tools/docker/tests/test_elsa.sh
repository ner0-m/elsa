#!/bin/bash

# saner programming env: these switches turn some bugs into errors
set -e -o errexit -o pipefail -o noclobber -o nounset

# Short and long options
OPTIONS=h
LONGOPTS=,build-cuda,build-quickvec,build-pybinds,build-coverage,build-docs,build-asan,help

# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi

# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

usage()
{
    printf "Usage: %s [<options>]\n" $(basename $0)
    echo ""
    echo "Available options:"
    echo "       --build-pybinds    Flag to activate generation of python bindings while building elsa"
    echo "       --build-coverage   Add seperate build from default build, which generates test coverage"
    echo "       --build-docs       Flags to exclusivly build documentation"
    echo "       --build-asan       Add seperate build from default build, which uses Address sanitizer"
    echo "       --build-cuda       Turn elsa flag for CUDA projectors"
    echo "       --build-quickvec   Turn elsa flag for using Quickvec and clang as a CUDA compiler"
    echo "  -h | --help             Display this help"
}

compiler=-

build_pybinds=n
build_docs=n
build_coverage=n
build_asan=n
build_cuda=n
build_quickvec=n

# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        --build-pybinds)
            build_pybinds=y
            shift
            ;;
        --build-docs)
            build_docs=y
            shift
            ;;
        --build-coverage)
            build_coverage=y
            shift
            ;;
        --build-asan)
            build_asan=y
            shift
            ;;
        --build-cuda)
            build_cuda=y
            shift
            ;;
        --build-quickvec)
            build_quickvec=y
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

if [ "${compiler}" == "-" ]; then
    echo "-- Check environment for compilers (prefer environment variables, over clang, over GCC)"

    # The double dash is needed so we can print a double dash...I LOOOOVE bash
    printf -- "-- Search for C Compiler..."
    if [[ ! -z "${CC:-}" ]]; then
        cc=$CC
    elif which clang > /dev/null 2>&1; then
        cc=clang
    elif which gcc > /dev/null 2>&1; then
        cc=gcc
    else
        echo "Not found (clang or gcc are not in '$PATH')"
        exit 1
    fi
    printf "Found: $cc\n"

    # Find our C++ compiler (clang++ or g++)
    printf -- "-- Search for C++ Compiler..."
    if [ ! -z "${CXX:-}" ]; then
        cxx=$CXX
    elif which clang++ > /dev/null 2>&1; then
        cxx=clang++
    elif which g++ > /dev/null 2>&1; then
        cxx=g++
    else
        echo "Not found (clang or gcc are not in '$PATH')"
        exit 1
    fi
    printf "Found: $cxx\n"
fi

# CMake flags, we'll need at some point, so make them available
CMAKE_FLAGS="-GNinja"

# If we use clang, be sure to use libc++ and libc++abi
if eval ${cxx} --version | grep -q clang ; then
    CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_CXX_FLAGS=\"-stdlib=libc++\" -DCMAKE_EXE_LINKER_FLAGS=\"-lc++abi\""
fi

if [ "$build_cuda" == "y" ]; then
    CMAKE_FLAGS="${CMAKE_FLAGS} -DELSA_BUILD_CUDA_PROJECTORS=ON"
fi

if [ "$build_quickvec" == "y" ]; then
    CMAKE_FLAGS="${CMAKE_FLAGS} -DELSA_CUDA_VECTOR=ON"
fi

# If we run asan or coverage these flags are appended
CMAKE_ASAN_FLAGS="-DELSA_BUILD_PYTHON_BINDINGS=OFF -DELSA_BUILD_CUDA_PROJECTORS=OFF -DCMAKE_BUILD_TYPE=Debug -DELSA_SANITIZER=\"Address;Undefined\""

# For coverage,
CMAKE_COVERAGE_FLAGS="-DCMAKE_BUILD_TYPE=Debug -DELSA_COVERAGE=ON -DELSA_BUILD_PYTHON_BINDINGS=OFF"

# If we build python bindings, we need to find llvm-config
if [ "$build_pybinds" == "y" ]; then
    printf -- "-- Search for llvm-config..."
    if llvm-config --version > /dev/null ; then
        echo "Found"
    else
        echo "Not found (llvm-config is not in PATH)"
        exit 1
    fi
fi

printf -- "-- Clone elsa to /tmp/elsa/..."
git clone https://gitlab.lrz.de/IP/elsa.git /tmp/elsa/ > /dev/null 2>&1
printf "done\n"


mkdir -p /tmp/elsa/build && cd /tmp/elsa/build/

printf "Running: cmake .. ${CMAKE_FLAGS}\n\n"
cmake .. ${CMAKE_FLAGS}

# Build only docs
if [ "$build_docs" == "y" ]; then
    echo "Build documentation..."
    ninja docs
    echo "...done"
    exit 0
fi

echo "Build elsa and test it..."
# build default
ninja
ninja build-examples
ninja tests

if [ "$build_coverage" == "y" ]; then
    echo "Build with test coverage..."
    mkdir -p /tmp/elsa/buildcov && cd /tmp/elsa/buildcov/

    echo "Running \"cmake .. ${CMAKE_FLAGS} ${CMAKE_COVERAGE_FLAGS}\""
    cmake .. ${CMAKE_FLAGS} ${CMAKE_COVERAGE_FLAGS}
    ninja
    ninja tests
    ninja test_coverage
    echo "...done"
fi

if [ "$build_asan" == "y" ]; then
    echo "Build elsa with Address sanitizer..."
    mkdir -p /tmp/elsa/buildasan && cd /tmp/elsa/buildasan/

    echo "Running \"cmake .. ${CMAKE_FLAGS} ${CMAKE_ASAN_FLAGS}\""
    cmake .. ${CMAKE_FLAGS} ${CMAKE_ASAN_FLAGS}
    ninja tests
    echo "...done"
fi

