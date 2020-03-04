Our automated build system is based on [GitLab CI/CD](https://docs.gitlab.com/ee/ci/).

Basically, we provide on a [.gitlab-ci.yml](https://gitlab.lrz.de/IP/elsa/blob/master/.gitlab-ci.yml) file, which describes all the different configurations we want to run. Most of the builds are run using [Docker containers](https://www.docker.com/resources/what-container). Only the parts, that have to be run localy are run using a shell executor (like documentation deployment).

#### Pipeline

The first part of our pipeline are checks for our [CONTRIBUTING.md](https://gitlab.lrz.de/IP/elsa/blob/master/CONTRIBUTING.md#style-guide).

Afterwards, we get to the build stage. The build stage is currently configured for a Clang 9, a GCC 9, a plain Ubuntu 18.04 (GCC 7.4) and a CUDA (GCC 7.4) image. With the Ubuntu image, we want to be sure, that our image compiles with a vanilla Ubuntu and only packages from the repository.

The test stage run our Unit test suite, but also some tests to ensure elsa is installed correctly and usable afterwards.

On the master branch we have the further stages of code coverage and dockumentation deployment.