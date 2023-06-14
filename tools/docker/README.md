# Docker images

This is a short walkthrough to keep track of all the docker images we're having and in which configuration

## How to build all images

Execute

> ./buildDockerContainers.sh

See `./buildDockerContainers.sh --help` for possible configuration options.

## How to build a single image

Run the command:

> docker build -t &lt;namespace&gt;/&lt;image&gt;:&lt;version&gt; -f &lt;docker-file&gt; [--build-arg &lt;arguments&gt;...] path/to/elsa/tools/docker

The test script assume the image to be in the namespace `staging`, then the image can be tested using

> ./testDriver.sh [&lt;options&gt;] -- &lt;image&gt; &lt;version&gt; &lt;script&gt; [&lt;arguments to test script&gt;]

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
ubuntu based image. this should be a vanilla version with only default packages, and therefore has CMake &lt;3.18.2&gt;

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

> ./testDriver.sh [&lt;options&gt;] -- &lt;image&gt; &lt;version&gt; &lt;script&gt; [&lt;arguments to test script&gt;]

Options include the docker network mode, privileged mode and if cuda should be enabled. The script then calls

> docker run [&lt;options&gt;] -t staging/&lt;image&gt;:&lt;version&gt; bash /tmp/tests/&lt;script&gt; [&lt;arguments to test script&gt;]

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
  - `--build-cuda`, set CMake flag `WANT_CUDA`
  - `--build-quickvec`, set CMake flag `ELSA_CUDA_VECTOR=ON`
  - `--build-docs`, don't build elsa, just it's documentation


## Docker registry

There is a docker registry running on the VM, which can be accessed (if rights are granted) at the
ssh address: `vmlasser7.in.tum.de`. The registry is accessible only if you're in the VPN of the
group, contact Tobias Lasser or one of the maintainers for more informations. The URL under which
it can be access is `docker.ciip.in.tum.de` at the port 5000.

If you want to pull or push images to the registry, you need to login first: using the following
commands:

```bash
docker login docker.ciip.in.tum.de # you'll need credentials
docker tag elsa/gcc:10 docker.ciip.in.tum.de/<namespace>/<image>:<version>
docker push docker.ciip.in.tum.de/<namespace>/<image>:<version>
docker pull docker.ciip.in.tum.de/<namespace>/<image>:<version>
docker logout
```

All machines used to run CI jobs, need the following line in the `/etc/docker/daemon.json`:
`"insecure-registries" : ["docker.ciip.in.tum.de:5000"]`. Also all runners also contain the following
lines in the `[runners.docker]` section:

* `allowed_images = ["elsa/*:*", "docker.ciip.in.tum.de:5000/elsa/*:*"]`
* `pull_policy = ["if-not-present", "always"]`

The CI has a pipeline variable `DOCKER_AUTH_CONFIG`, which enables the authentication to the registry
inside of the CI jobs. [This guide](https://docs.gitlab.com/ce/ci/docker/using_docker_images.html#use-statically-defined-credentials)
was used to set it up, once the docker registry was up and running.

#### Setup process

The following guides were used to setup the docker registry:

* https://docs.docker.com/registry/deploying/
* https://docs.docker.com/registry/insecure/
* https://docs.gitlab.com/ce/ci/docker/using_docker_images.html#use-statically-defined-credentials

First, credentials are generated using the following command:

```bash
docker run --entrypoint htpasswd httpd:2 -Bbn <username> <passphrase> > auth/htpasswd
```

Then the docker registry is setup

```bash
docker run -d \
    --restart=always --name elsa-registry -p 5000:5000 \
    -v "$(pwd)"/auth:/auth \
    -e "REGISTRY_AUTH=htpasswd" \
    -e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
    -e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
    -v /var/lib/rbg-cert/live/:/certs \
    -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/host:intum:vmlasser7.cert.pem \
    -e REGISTRY_HTTP_TLS_KEY=/certs/host:intum:vmlasser7.privkey.pem \
    registry:2
```

This command sets up a container with the name `elsa-registry`, which is always started, if the docker
daemon is started. It exposes itself to port 5000. According to the [this guide](https://docs.docker.com/registry/deploying/)
the credentials are mounted inside the container and setup.

Also the certificate is mounted and setup, such that a secure TLS connection can be setup.
However, adding `"insecure-registries" : ["docker.ciip.in.tum.de:5000"]` to `/etc/docker/daemon.json`
is necessary to accept the certificate.

##### Initial push to registry

If all images are present at some machine, you can push them to the registry (if for some reason it's
setup newly), with a similar command to this:

```bash
docker images | grep ^elsa | tr -s ' ' | cut -d ' ' -f 1-2 | sed 's/ /:/g' | \
xargs -I {} echo "sudo docker tag {} docker.ciip.in.tum.de:5000/{} && sudo docker push docker.ciip.in.tum.de:5000/{}" | \
cat | bash
```

Don't forget to login first and then logout

##### Stop registry

```bash
docker container stop elsa-registry && sudo docker container rm -v elsa-registry
```

##### Remove registry

```bash
docker container rm -v elsa-registry
```
