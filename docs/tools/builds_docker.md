Looking into the `tools/docker` folder, you'll find our dockerfiles, a scripts folder, a tests folder and a build scripts. The build scripts builds all images into a staging area, runs some tests on the images and then tags them into production and removes the staging tag. The tests range from simple calls to e.g. CMake or for the compiler images, we try to compile elsa and run its tests.

#### Images

This section gives a small overview of the important images and some information, that should help maintaining them.

##### Overview

| Image name           | Compiler         | CMake  | Intel DNNL | CUDA | Others                                                    |
|----------------------|------------------|--------|------------|------|-----------------------------------------------------------|
| elsa/gcc             | gcc 6, 7, 8, 9   | 3.16.4 | 1.1.1      | No   |                                                           |
| elsa/clang           | clang 6, 7, 8, 9 | 3.16.4 | 1.1.1      | No   |                                                           |
| elsa/ubuntu          | gcc 7.4          | 3.10.2 | 1.1.1      | No   |                                                           |
| elsa/cuda            | gcc, clang 8     | 3.16.4 | 1.1.1      | 9.2, 10.0 10.2 | lcov                                                      |
| elsa/documentation   | gcc 7.4          | 3.16.4 | No         | No   | Doxygen, Sphinx, Breathe, M2R, Read the Docs Sphinx Theme |
| elsa/static-analysis | gcc 7.4          | 3.16.4 | No         | No   | clang-tidy (Version 8), clang-format (Version 8)          |

All of the images are currently based on Ubuntu 18.04.

Two images not mentioned are the base and CMake images. All images, which have CMake 3.16.4 are based on it (exept the CUDA image, that is generated independently of all the others)


##### GCC

All GCC versions are pulled from the [PPA for Ubuntu Toolchain Uploads](https://launchpad.net/~ubuntu-toolchain-r/+archive/ubuntu/test). The PPA is added and the wanted version of GCC is installed using `apt-get`.

##### Clang

The install script for the clang compiler is heavely based on [install-clang](https://github.com/rsmmr/install-clang). It builds clang in a 3 step way. It builds clang, then builds itself again (bootstraping it) and libc++. And in a final step it bootstrap itself again now using libc++ instead of libstdc++.

This is a rather involved work and takes quite some time to build. We should investiage if other solutions are viable.

##### Ubuntu

This image is a baseline, as we want our library to work with installed packages from Ubuntu 18.04. Therefore, only default packages are used, no other packages are installed via some third party or anything.

##### CUDA

This image is a little different from the others, as it is based on the [CUDA dockerfile](https://hub.docker.com/r/nvidia/cuda/). Therefore, everything is installed as in the other images, but instaed of an inheritance structure, it is plainly installed again. This is not perfectly, but it works for now.

It also contains lcov, as we run our test coverage on the CUDA image. Anyway we could not cover all our sources.


##### Static Analysis

This was decoupled from the other images and mostly contains clang-tidy and clang-format.

#### Build images

All images can be build by running the `buildDockerContainers.sh` script from within the `tools/docker` folder. It will build all images [listed](#overview). Run it with `-h` or `--help` to see all possible configuration on the command line. It should include the used CMake version and DNNL version.

Individual images should be build from the folder. Then you can run

```
docker build -t elsa/image:tag -f MyDocFile .
```

To reduce used space, I would also recommend running `docker system prune --volumes` after builds. This will remove (among others) all dangling images, that were created during the build stages. (This also deletes unused containers, which might not be, what you want!) Further many images rely on build arguments to choose the specific version. Usually they have a default value. Please check the `buildDockerContainers.sh` for specific values. To specify them add `--build-arg MY_VARIABLE="my-value"` to the build command above.

All images have the tests scripts in `/tmp/tests/`, So to compile and run tests on elsa run:

```
docker run -it elsa/image:tag /tmp/tests/test_elsa.sh
```

This will pull elsa, build it and run it's tests.