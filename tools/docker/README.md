# Docker images

This is a short walkthrough to keep track of all the docker images we're having and in which configuration
 
## How to build all images 
 
Execute

> ./buildDockerContainers.sh
 
See `./buildDockerContainers.sh --help` for possible configuration options. 
 
## How to build a single image 

Run the command:

> docker build -t <namespace>/<image>:<version> -f <docker-file> [--build-arg <arguments>...] path/to/elsa/tools/docker

The test script assume the image to be in the namespace `staging`, then the image can be tested using

> ./testDriver.sh [<options>] -- <image> <version> <script> [<arguments to test script>]
 
See the section at the bottom. 

## Overview of docker images

- `elsa/clang-format`, contains clang-format 
- `elsa/clang-tidy`, build ontop of elsa/cuda-clang, plus clang-tidy 
- `elsa/clang`, base image plus CMake plus clang compiler 
- `elsa/clang-pybinds`, clang image, plus packages for python bindings 
- `elsa/gcc`, base image plus CMake plus GNU compiler
- `elsa/gcc-pybinds`, gcc image, plus packages for pythong bindings
- `elsa/ubuntu`, vanilla Ubuntu version (only Ubuntu packages basically, no extra packages or source builds)
- `elsa/cuda`, based on [official nvidia/cuda image](https://hub.docker.com/r/nvidia/cuda/), plus our base packages and CMake
- `elsa/cuda-clang`, CUDA with clang as compiler instead of NVCC
- `elsa/coverage`, cuda image, plus lcov

All images that include an compiler also install oneDNN (currently version 1.6.2). To not rely on ABI, this is only
installed once the final compiler for the image is installed and the enviorment variables are set. oneDNN is install in
`elsa/clang`, `elsa/clang-pybinds`, `elsa/gcc`, `elsa/gcc-pybinds`, `elsa/cuda`, `elsa/cuda-clang` and `elsa/ubuntu`.
All images that build on these, also contain oneDNN.
 
All images that contain CMake install version 3.18.2 and all images installing ninja install ninja 1.10.1. Expect the
ubuntu based image. this should be a vanilla version with only default packages, and therefore has CMake <FIIIILLL me>

## Overview of docker files
 
The arguments can be set via `--build-arg` in the `docker build` call. If no default is given, it is required to be set.
 
### DocFileBase

**file for `elsa/base`, versions *18.04 and 20.04***
 
Base image which is build ontop of Ubuntu (currently both 18.04, and 20.04) and it includes packages, which are 
required pretty much everywhere. These include git, wget, ninja (currently 1.10.1), python3, numpy and some others.
 
##### Arguments

- UBUNTU_VERSION (default 20.04)
 
### DocFileBaseCMake

**file for `elsa/cmake`, versions *18.04 and 20.04***

Image build ontop of `elsa/base`, install CMake (currently version 3.18.2) to /tmp/cmake-install. This is used to copy
the folder into the other docker images. This way, we don't need to build CMake from source for all of them. This is
not a super clean solution and only works, if the same packages are present and the same OS is usued and even then,
we might silently rely on some things. But for now it works.
 
##### Arguments

- UBUNTU_VERSION (default 20.04)
- CMAKE_VERSION
- CMAKE_VERIONS_PATH
 
### DocFileClang

**file for `elsa/clang`, versions *9 and 10***

Image for clang compiler. Build ontop of `elsa/base:20.04`, uses the CMake install from `elsa/cmake:20.04` and installs
clang from the Ubuntu repositories.
 
##### Arguments

- COMPILER_VERSION
- ONEDNN_VERSION (default 1.6.2) 
 
### DocFilePyBinds 
 
**file for `elsa/clang-pybinds`, versions *9 and 10***
 
Image that includes packages to generate python bindings (TODO: here is a potential
for optimization in docker size. It could be there are currently installed to many packages)
It builds on top of either `elsa/gcc` or `elsa/clang` to build python bindigs with both compilers.
 
##### Arguments

- COMPILER_VERSION
- IMAGE
- LLVM_VERSION 
 
### DocFileClangFormat 
 
**file for `elsa/clang-format`, versions *8, 9 and 10***

Image for clang-format. Build ontop of the official `ubuntu:20.04` image, only install clang-format.
 

##### Arguments
 
- CLANG_FORMAT_VERSION (default 10) 
 
### DocFileClangTidy 
 
**file for `elsa/clang-tidy`, versions *8, 9 and 10***

Image for clang-tidy. Build ontop of the `elsa/cuda-clang:20.04` image (to make sure, all files are checked with 
clang-tidy), only install clang-format.
 
##### Arguments
 
- CLANG_TIDY_VERSION (default 10) 
 
### DocFileCoverage 
 
**file for `elsa/coverage`, versions *10.2 and 11.0***
 
Image containing packages for coverage (such as lcov). Builds ontop of `elsa/cuda` image. 
 
##### Arguments
 
No arguments 
 
### DocFileCuda 
 
**file for `elsa/cuda`, versions *10.2 and 11.0***
 
Image build ontop of `nvidia/cuda`, contains our base packages CMake 
 
##### Arguments
 
- CUDA_VERSION (default 10.2) 
- UBUNTU_VERSION (default 18.04) 
- GCC_VERSION (default 8) 
- ONEDNN_VERSION (default 1.6.2) 
 
### DocFileCudaWithClang
 
**file for `elsa/cuda-cuda`, versions *10.0***

Image that builds clang from scratch, with support to compile CUDA code. Contains CMake and our base packages. As
versining is very specific, we rely on CUDA 10.0 and clang 8.0.1.
 
##### Arguments
 
- CUDA_VERSION (default 10.0) 
- UBUNTU_VERSION (default 18.04) 
- CLANG_VERSION (default 8.0.1) 
- ONEDNN_VERSION (default 1.6.2) 
 
### DocFileDocumentation

**file for `elsa/documentation`**
 
File containing all our packages needed to build documentation, but not currently used in the CI Pipeline. 
 
##### Arguments
 
No arguments 

### DocFileGcc 
 
**file for `elsa/gcc`, versions *9 and 10***
 
Image for GNU compiler. Build ontop of `elsa/base:20.04`, uses the CMake install from `elsa/cmake:20.04` and installs
GCC from the Ubuntu repositories. Very similar to the clang image.
 
##### Arguments
 
- UBUNTU_VERSION (default 20.04)
- COMPILER_VERSION 
- ONEDNN_VERSION (default 1.6.2) 
 
### DocFileUbuntu

**file for `elsa/ubuntu`, versions *18.04 and 20.04***
 
Vanilla Ubuntu image, all packages are from the package repositories and no extra packages are added. 
 
##### Arguments
 
- UBUNTU_VERSION (default 20.04)
- ONEDNN_VERSION (default 1.6.2) 

## Overview of install scripts
 
There are 3 install scripts in the `scripts` folder. Most scripts have the command `set -eux` at the top, this shows
the commands while running the scripts and exits when an error occures. And I'm not a good bash scripts writer, so
if there are any better ways to do it, feel free to change it.

- `install_base.sh`, installs the base images, creates some symlinks for python3 and pip3 and builds ninja from source
- `install_cmake.sh`, install CMake to `/tmp/cmake-install/` in the docker image 
- `install_intel-oneDNN.sh` installs oneDNN into `/tmp/onednn_install` in the docker image

## Overview of test scripts 
 
All tests are executed using the `testDriver.sh` file. This is not part of the docker images, but it takes care of
some post and pre work before running the actual test script. Call it with

> ./testDriver.sh [<options>] -- <image> <version> <script> [<arguments to test script>]
 
Options include the docker network mode, privileged mode and if cuda should be enabled. The script then calls

> docker run [<options>] -t staging/<image>:<version> bash /tmp/tests/<script> [<arguments to test script>]
 
If the tests are run succesfully, the image will be transfered to `elsa/<image>:<version>`. 
 
All images have their own test scripts, and if it's a compiler image, then elsa will be build as a test. 

- `test_base.sh` checks if base packages are accessible
- `test_clang-format.sh` checks that the wanted clang-format version is present (version expected as first argument)
- `test_clang-tidy.sh` similar to `test_clang-format.sh`, checks for the clang-tidy version (version expected as first argument)
- `test_cmake.sh`, check if cmake is present in its tmp folder
- `test_elsa.sh`, build elsa with the given image. Depending on options it will build elsa additionally build a different config.
  - `--build-pybinds` will activate the genration of python bindings and check that `llvm-config` is present
  - `--build-coverage`, additionally build the test coverage
  - `--build-asan`, additionally build elsa with Address Sanitizer activated 
  - `--build-cuda`, set CMake flag `ELSA_BUILD_CUDA_PROJECTORS` 
  - `--build-quickvec`, set CMake flag `ELSA_CUDA_VECTOR=ON` 
  - `--build-docs`, don't build elsa, just it's documentation
