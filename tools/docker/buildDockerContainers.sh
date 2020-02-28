#!/bin/bash

# saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

### Argument parsing ###

# based on https://stackoverflow.com/a/29754866

# -allow a command to fail with !’s side effect on errexit
# -use return value from ${PIPESTATUS[0]}, because ! hosed $?
! getopt --test > /dev/null 
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

# our options (see usage)
OPTIONS=n:c:C:b:B:d:u:th
LONGOPTS=network:,cuda-version:,cuda-clang-version:,cmake-version:,cmake-patch-version:dnnl-version:,ubuntu-version:,no-tidy,help

# -regarding ! and PIPESTATUS see above
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

# default variables
clang_version=9.0.1
cuda_version=10.2
cmake_version=3.16
cmake_patch_version=4
dnnl_version=1.1.1
ubuntu_release=18.04
network_mode=-
do_cleanup=y
usage()
{
    printf "Usage: %s [<options>]\n" $(basename $0)
    echo ""
    echo "Available options:"
    echo "  -c | --cuda-clang-version  <x.y.z>  Clang version for CUDA image [default $clang_version]"
    echo "  -C | --cuda-version          <x.y>  CUDA version [default $cuda_version]"
    echo "  -b | --cmake-version         <x.y>  CMake major version to use in images [default $cmake_version]"
    echo "  -B | --cmake-patch-version     <x>  CMake patch version to use in images [default $cmake_patch_version]]"
    echo "  -d | --dnnl-version        <x.y.z>  Intel DNNL version to use in all images [default $dnnl_version]"
    echo "  -u | --ubuntu-version        <x.y>  Ubuntu verison for dedicated Ubuntu image [default: ${ubuntu_release}]"
    echo "  -n | --network               mode   Network mode passed to docker command [host, bridge]"
    echo "  -t | --no-tidy                      Don't clean up dangling docker images"
    echo "  -h | --help                         Display this help"
    echo " with x,y,z being numbers (the exact number has to be provided)"
}

# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -c|--cuda-clang-version)
            clang_version="$2"
            shift 2
            ;;
        -C|--cuda-version)
            cuda_version="$2"
            shift 2
            ;;
        -b|--cmake-version)
            cmake_version="$2"
            shift 2
            ;;
        -B|--cmake-patch-version)
            cmake_patch_version="$2"
            shift 2
            ;;
        -d|--dnnl-version)
            dnnl_version="$2"
            shift 2
            ;;
        -u|--ubuntu-version)
            ubuntu_release="$2"
            shift 2
            ;;
        -n|--network)
            network_mode="$2"
            shift 2
            ;;
        -t|--no-tidy)
            do_cleanup=n
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

docker_arguments=""

if docker network ls | grep -o "$network_mode" | sort | uniq > /dev/null ; then
	echo "Network mode: $network_mode"
    docker_arguments="$docker_arguments --network=$network_mode"
fi

echo "=================="
echo "==== Building images with: "
echo "====== CMake version: ${cmake_version}.${cmake_patch_version}"
echo "====== DNNL version: ${dnnl_version}"
echo "====== Ubuntu release: ${ubuntu_release}"
echo "====== Cuda version: ${cuda_version}, with clang version ${clang_version}"
echo "=================="

# so we can read this message sleep a little
sleep 5

### Build base image ###

echo "Building docker image for Base..."
docker build $docker_arguments -t elsa/base:latest -f DocFileBase .
echo "... building Base done."

### Build CMake image ###

CMAKE_IMAGE=base-cmake:$cmake_version.$cmake_patch_version

echo "Building image ${CMAKE_IMAGE}..."
docker build $docker_arguments -t staging/${CMAKE_IMAGE} -f DocFileBaseCMake \
             --build-arg CMAKE_VERSION=${cmake_version} --build-arg CMAKE_VERSION_PATCH=${cmake_patch_version} . 
echo "...done."

./tests/test_template.sh "base-cmake" "${cmake_version}.${cmake_patch_version}" "test_cmake.sh" --network=$network_mode

### Build Static Analysis image ###

echo "Building docker image static-analysis..."
docker build $docker_arguments -t staging/static-analysis:8 -f DocFileStaticAnalysis .
echo "...done."

./tests/test_template.sh "static-analysis" "8" "test_static-analysis.sh" --network=$network_mode

### Build clang images ###

echo "Docker arguments: $docker_arguments"

echo "Building docker image for Clang 6..."
docker build $docker_arguments -t staging/clang:6 -f DocFileClangBase --build-arg COMPILER_VERSION="6.0.1" --build-arg DNNL_VERSION=${dnnl_version} .
echo "... building Clang 6 done."

./tests/test_template.sh clang 6 test_compile.sh --network=$network_mode

echo "Building docker image for Clang 7..."
docker build $docker_arguments -t staging/clang:7 -f DocFileClangBase --build-arg COMPILER_VERSION="7.1.0" --build-arg DNNL_VERSION=${dnnl_version} .
echo "... building Clang 7 done."

./tests/test_template.sh clang 7 test_elsa.sh --network=$network_mode

echo "Building docker image for Clang 8..."
docker build $docker_arguments -t staging/clang:8 -f DocFileClangBase --build-arg COMPILER_VERSION="8.0.1" --build-arg DNNL_VERSION=${dnnl_version} .
echo "... building Clang 8 done."

./tests/test_template.sh clang 8 test_elsa.sh --network=$network_mode

echo "Building docker image for Clang 9..."
docker build $docker_arguments -t staging/clang:9 -f DocFileClangBase --build-arg COMPILER_VERSION="9.0.1" --build-arg DNNL_VERSION=${dnnl_version} .
echo "... building Clang 9 done."

./tests/test_template.sh clang 9 test_elsa.sh --network=$network_mode

### Build gcc images ###

echo "Building docker image for gcc 6..."
docker build $docker_arguments -t staging/gcc:6 -f DocFileGcc --build-arg COMPILER_VERSION=6 --build-arg DNNL_VERSION=${dnnl_version} .
echo "... building gcc 6 done."

./tests/test_template.sh gcc 6 test_compile.sh --network=$network_mode

echo "Building docker image for gcc 7..."
docker build $docker_arguments -t staging/gcc:7 -f DocFileGcc --build-arg COMPILER_VERSION=7 --build-arg DNNL_VERSION=${dnnl_version} .
echo "... building gcc 7 done."

./tests/test_template.sh gcc 7 test_elsa.sh --network=$network_mode

echo "Building docker image for gcc 8..."
docker build $docker_arguments -t staging/gcc:8 -f DocFileGcc --build-arg COMPILER_VERSION=8 --build-arg DNNL_VERSION=${dnnl_version} .
echo "... building gcc 8 done."

./tests/test_template.sh gcc 8 test_elsa.sh --network=$network_mode

echo "Building docker image for gcc 9..."
docker build $docker_arguments -t staging/gcc:9 -f DocFileGcc --build-arg COMPILER_VERSION=9 --build-arg DNNL_VERSION=${dnnl_version} .
echo "... building gcc 9 done."

./tests/test_template.sh gcc 9 test_elsa.sh --network=$network_mode

### Build Cuda image ###

# Tests for cuda images always include the cov argument, to indicate we want to also build the test coverage

echo "Building docker image for Cuda 9.2..."
docker build $docker_arguments -t staging/cuda:9.2 -f DocFileCuda --build-arg CUDA_VERSION=9.2 \
             --build-arg CLANG_VERSION="8.0.1" --build-arg DNNL_VERSION=${dnnl_version}  \
             --build-arg CMAKE_VERSION=${cmake_version} --build-arg CMAKE_VERSION_PATCH=${cmake_patch_version} .
echo "... building Cuda 9.2 done."

./tests/test_template.sh --network=$network_mode cuda "9.2" test_elsa.sh cov

echo "Building docker image for Cuda ${cuda_version}..."
docker build $docker_arguments -t staging/cuda:${cuda_version} -f DocFileCuda --build-arg CUDA_VERSION=${cuda_version} \
             --build-arg CLANG_VERSION=${clang_version} --build-arg DNNL_VERSION=${dnnl_version}  \
             --build-arg CMAKE_VERSION=${cmake_version} --build-arg CMAKE_VERSION_PATCH=${cmake_patch_version} .
echo "... building Cuda ${cuda_version} done."

./tests/test_template.sh --network=$network_mode cuda "${cuda_version}" test_elsa.sh cov

### Build Ubuntu image ###

echo "Building docker image for Ubuntu ${ubuntu_release}..."
docker build $docker_arguments -t staging/ubuntu:${ubuntu_release} -f DocFileUbuntu \
             --build-arg UBUNTU_VERSION=${ubuntu_release} --build-arg DNNL_VERSION=${dnnl_version} .
echo "... building Ubuntu ${ubuntu_release} done."

./tests/test_template.sh ubuntu "${ubuntu_release}" test_elsa.sh --network=$network_mode

echo "Building docker image for Documentation..."
docker build $docker_arguments -t staging/documentation:latest -f DocFileDocumentation .
echo "... building Documentation image done."

# last argument indicates, we want to build the documentation
./tests/test_template.sh --network=$network_mode documentation latest test_elsa.sh doc

### Cleanup ##

# remove dangling contaienrs, images and volumes

if [[ "$do_cleanup" == "y" ]]; then
    echo "Cleaning up..."
    docker system prune --volumes --force
    echo "done!"
fi
