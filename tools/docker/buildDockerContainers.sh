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
LONGOPTS=network:,cuda-version:,cuda-clang-version:,cmake-version:,cmake-patch-version:onednn-version:,ubuntu-version:,no-tidy,help

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
clang_version=8.0.1
static_analysis_version=8 
cuda_version=10.0
cmake_version=3.18
cmake_patch_version=2
onednn_version=1.6.2
ubuntu_release=20.04
network_mode=-
do_cleanup=y
build_legacy_gcc=n
build_legacy_clang=n
usage()
{
    printf "Usage: %s [<options>]\n" $(basename $0)
    echo ""
    echo "Available options:"
    echo "  -c | --cuda-clang-version  <x.y.z>  Clang version for CUDA image [default $clang_version]"
    echo "  -C | --cuda-version          <x.y>  CUDA version [default $cuda_version]"
    echo "  -b | --cmake-version         <x.y>  CMake major version to use in images [default $cmake_version]"
    echo "  -B | --cmake-patch-version     <x>  CMake patch version to use in images [default $cmake_patch_version]]"
    echo "  -d | --onednn-version      <x.y.z>  Intel oneDNN version to use in all images [default $onednn_version]"
    echo "  -u | --ubuntu-version        <x.y>  Ubuntu verison for dedicated Ubuntu image [default: ${ubuntu_release}]"
    echo "  -n | --network               mode   Network mode passed to docker command [host, bridge]"
    echo "  -t | --no-tidy                      Don't clean up dangling docker images"
    echo "  -s | --static-analysis         <x>  Versions for clang-tidy and clang-format" 
    echo "  -h | --help                         Display this help"
    echo "     | --build-legacy-gcc             Build images for old GCC versions (which include gcc 6,7,8)"
    echo "     | --build-legacy-clang           Build images for old clang versions (which include clang 6,7,8)"
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
        -d|--onednn-version)
            onednn_version="$2"
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
        -s|--static-analysis)
            do_cleanup=n
            shift 
            ;;
        --build-legacy-gcc)
            build_legacy_gcc=y 
            shift 
            ;; 
        --build-legacy-clang)
            static_analysis_version="$2"
            shift 2
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
echo "====== oneDNN version: ${onednn_version}"
echo "====== Ubuntu release: ${ubuntu_release}"
echo "====== Gpu only (CUDA with clang), CUDA versions: ${cuda_version}, with clang version ${clang_version}"
echo "=================="

# so we can read this message sleep a little
sleep 5

### Build base image ###

# Base images have to be build for Ubuntu 20.04 and 18.04, for as long as we have Ubuntu 18.08 in our cuda images
# We base the cmake images of from these and therefore, copy the CMake installs into our CUDA images. That we don't
# silently depend on some release specific libraries or anything, we have to be careful here.
tmp_build=18.04
echo "Building docker image for Base with Ubuntu ${tmp_build}..."
docker build $docker_arguments -t staging/base:${tmp_build} -f DocFileBase --build-arg UBUNTU_VERSION=${tmp_build} .
echo "... building Base done."

./testDriver.sh --network=$network_mode "base" "${tmp_build}" "test_base.sh"
 
tmp_build=20.04
echo "Building docker image for Base with Ubuntu ${tmp_build}..."
docker build $docker_arguments -t staging/base:${tmp_build} -f DocFileBase --build-arg UBUNTU_VERSION=${tmp_build} .
echo "... building Base done."

./testDriver.sh --network=$network_mode "base" "${tmp_build}" "test_base.sh"

# ### Build CMake image ###

# Same as before until we support CUDA on Ubuntu 18.04 we need two images. Now the the version numbers of the CMake
# images denote the Ubuntu version, to make it easier down the line
tmp_build=18.04
echo "========"
echo "Building image cmake:${tmp_build}..."
docker build $docker_arguments -t staging/cmake:${tmp_build} -f DocFileBaseCMake \
             --build-arg UBUNTU_VERSION=${tmp_build} \
             --build-arg CMAKE_VERSION=${cmake_version} --build-arg CMAKE_VERSION_PATCH=${cmake_patch_version} .
echo "...done."

./testDriver.sh --network=$network_mode "cmake" "${tmp_build}" "test_cmake.sh"
 
tmp_build=20.04
echo "========"
echo "Building image cmake:${tmp_build}..."
docker build $docker_arguments -t staging/cmake:${tmp_build} -f DocFileBaseCMake \
             --build-arg UBUNTU_VERSION=${tmp_build} \
             --build-arg CMAKE_VERSION=${cmake_version} --build-arg CMAKE_VERSION_PATCH=${cmake_patch_version} .
echo "...done."

./testDriver.sh --network=$network_mode "cmake" "${tmp_build}" "test_cmake.sh"


### Build clang images ###

echo "Docker arguments: $docker_arguments"
 
echo "========" 
echo "Building clang images" 
ver_start=9 
ver_end=10 
if [[ "$build_legacy_clang" == "y" ]]; then
    # TODO: this will not work for now, as we still build python bindings in these images, and the packages don't
    # exist for these clang versions, so let's fix this later
    ver_start=6
fi

for ver in $(seq ${ver_start} ${ver_end})
do
    echo "Building docker image for Clang ${ver}..."
    docker build $docker_arguments -t staging/clang:${ver} -f DocFileClang --build-arg COMPILER_VERSION="${ver}" \
        --build-arg ONEDNN_VERSION=${onednn_version} --build-arg UBUNTU_VERSION=${ubuntu_release} .
    echo "... building Clang ${ver} done."

    ./testDriver.sh --network=$network_mode --privileged -- clang ${ver} test_elsa.sh --build-asan
done
 
echo "========" 
 
### clang with packages for python bindings ###
 
echo "========" 
echo "Building images with python binding related stuff" 
 
for ver in $(seq 9 10)
do
    echo "Building docker image for Clang ${ver} with Python bindings..."
    docker build $docker_arguments -t staging/clang-pybinds:${ver} -f DocFilePyBinds \
        --build-arg LLVM_VERSION="${ver}" --build-arg IMAGE="clang" --build-arg COMPILER_VERSION="${ver}" .
    echo "... building Clang ${ver} done."

    ./testDriver.sh --network=$network_mode --privileged -- "clang-pybinds" ${ver} test_elsa.sh \
        --build-pybinds
done
 
echo "========" 

### Build gcc images ###
 
ver_start=9 
ver_end=10 
if [[ "$build_legacy_gcc" == "y" ]]; then
    ver_start=7 
fi
 
echo "Build GCC images (versions ${ver_start} to ${ver_end}"
 
for ver in $(seq ${ver_start} ${ver_end})
do
    echo "Building docker image for gcc ${ver}..."
    docker build $docker_arguments -t staging/gcc:${ver} -f DocFileGcc --build-arg COMPILER_VERSION=${ver} \
        --build-arg ONEDNN_VERSION=${onednn_version} .
    echo "... building gcc ${ver} done."
     
    ./testDriver.sh --network=$network_mode --privileged -- "gcc" ${ver} "test_elsa.sh" --build-asan
done
 
# Only build python bindings for GCC 9 and 10 
for ver in $(seq 9 10)
do
    echo "Building docker image for GCC ${ver} with Python bindings..."
    # Build python bindings with the same LLVM version as we have GCC, this is a little arbitarly choosen, but works.
    docker build $docker_arguments -t staging/gcc-pybinds:${ver} -f DocFilePyBinds \
        --build-arg LLVM_VERSION="${ver}" --build-arg IMAGE="gcc" --build-arg COMPILER_VERSION="${ver}" .
    echo "... building Clang ${ver} done."

    ./testDriver.sh --network=$network_mode --privileged -- "gcc-pybinds" ${ver} test_elsa.sh \
        --build-pybinds
done
 
### Build Cuda image ###

# Cuda 10.2 only has a Ubuntu 18.04 Docker image in the official repo, so we're sticking with this version
# Cuda 10.2 only supports GCC up to version 8, sooooo we're stuck with it for now -.- 
tmp_ver="10.2" 
echo "Building docker image for Cuda ${tmp_ver}..." 
docker build $docker_arguments -t staging/cuda:${tmp_ver} -f DocFileCuda --build-arg CUDA_VERSION=${tmp_ver} \
    --build-arg UBUNTU_VERSION=18.04 --build-arg GCC_VERSION=8 --build-arg ONEDNN_VERSION=${onednn_version} \
    --build-arg CMAKE_VERSION=${cmake_version} --build-arg CMAKE_VERSION_PATCH=${cmake_patch_version} .
echo "... building Cuda ${tmp_ver} done."

./testDriver.sh --network=$network_mode --run-cuda --privileged -- cuda "${tmp_ver}" test_elsa.sh --build-asan --build-cuda

# Cuda 11.0 has Ubuntu 20.04 support 
tmp_ver="11.0" 
echo "Building docker image for Cuda ${tmp_ver}..." 
docker build $docker_arguments -t staging/cuda:${tmp_ver} -f DocFileCuda --build-arg CUDA_VERSION=${tmp_ver} \
    --build-arg UBUNTU_VERSION=20.04 --build-arg GCC_VERSION=9 --build-arg ONEDNN_VERSION=${onednn_version} \
    --build-arg CMAKE_VERSION=${cmake_version} --build-arg CMAKE_VERSION_PATCH=${cmake_patch_version} .
echo "... building Cuda ${tmp_ver} done."

# Don't test for CUDA 11 for now, as we don't support it yet officially 
# ./testDriver.sh --network=$network_mode  --run-cuda --privileged -- cuda "${tmp_ver}" test_elsa.sh --build-cuda
 
# Cuda with clang version (GPU-only image)
echo "Building docker image for Cuda with clang with CUDA ${cuda_version} and clang ${clang_version}..."
docker build $docker_arguments -t staging/cuda-clang:${cuda_version} -f DocFileCudaWithClang \
    --build-arg CUDA_VERSION=${cuda_version} --build-arg UBUNTU_VERSION=18.04  \
    --build-arg CLANG_VERSION=${clang_version} --build-arg ONEDNN_VERSION=${onednn_version}  \
    --build-arg CMAKE_VERSION=${cmake_version} --build-arg CMAKE_VERSION_PATCH=${cmake_patch_version} .
echo "... building Cuda ${cuda_version} done."

./testDriver.sh --network=$network_mode --run-cuda --privileged -- "cuda-clang" "${cuda_version}" "test_elsa.sh" \
    --build-quickvec
 

### Build Ubuntu image ###

echo "Building docker image for Ubuntu ${ubuntu_release}..."
docker build $docker_arguments -t staging/ubuntu:${ubuntu_release} -f DocFileUbuntu \
             --build-arg UBUNTU_VERSION=${ubuntu_release} --build-arg ONEDNN_VERSION=${onednn_version} .
echo "... building Ubuntu ${ubuntu_release} done."

./testDriver.sh --network=$network_mode ubuntu "${ubuntu_release}" test_elsa.sh
 
### Coverage image ### 
 
echo "Building coverage image..." 
docker build $docker_arguments -t staging/coverage:10.2 -f DocFileCoverage .
echo "... building coverage image done."
 
./testDriver.sh --network=$network_mode --run-cuda -- "coverage" "10.2" "test_elsa.sh" --build-coverage --build-cuda 
 
### Build Static Analysis image ###

echo "========" 
 
for ver in $(seq 8 10)
do
    echo "Building docker image for clang-format-${ver}..."
    docker build $docker_arguments -t staging/clang-format:${ver} -f DocFileClangFormat \
        --build-arg CLANG_FORMAT_VERSION=${ver} .
    echo "...done."

    ./testDriver.sh --network=$network_mode "clang-format" "${ver}" "test_clang-format.sh" "${ver}"

    # This has to be after the elsa/cuda-clang image
    echo "Building docker image for clang-tidy-${ver}..."
    docker build $docker_arguments -t staging/clang-tidy:${ver} -f DocFileClangTidy \
        --build-arg CLANG_TIDY_VERSION=${ver} --build-arg BASE_IMAGE="cuda-clang:10.0" .
    echo "...done."

    ./testDriver.sh --network=$network_mode "clang-tidy" "${ver}" "test_clang-tidy.sh" "${ver}"
 
    echo "========" 
done

### Documentation image ### 
 
echo "========" 
 
echo "Building docker image for Documentation..."
docker build $docker_arguments -t staging/documentation:latest -f DocFileDocumentation .
echo "... building Documentation image done."

# last argument indicates, we want to build the documentation
# ./testDriver.sh --network=$network_mode -- documentation latest test_elsa.sh --build-docs
 
echo "========" 

### Cleanup ##

# remove dangling contaienrs, images and volumes

if [[ "$do_cleanup" == "y" ]]; then
    echo "Cleaning up..."
    docker system prune --volumes --force
    echo "done!"
fi
