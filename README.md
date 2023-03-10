# elsa - an elegant framework for tomographic reconstruction

**elsa** is an operator- and optimization-oriented framework for tomographic
reconstruction, with a focus on iterative reconstruction algorithms.
It is usable from Python and C++.

By design, **elsa** provides a flexible description of multiple imaging modalities.
The current focus is X-ray based computed tomography (CT) modalities such as
attenuation X-ray CT, phase-contrast X-ray CT based on grating interferometry
and (anisotropic) Dark-field X-ray CT. Other imaging modalities can be
supported easily and are usable with our extensive suite of optimization algorithms.

CUDA implementations for the computationally expensive forward models, which
simulate the physical measurement process of the imaging modality, are available
in **elsa**.

The framework is mostly developed by the Computational Imaging and Inverse Problems
(CIIP) group at the Technical University of Munich. For more info about our research
checkout our at [ciip.cit.tum.de/](https://ciip.cit.tum.de/).

The source code of **elsa** is hosted at
[gitlab.lrz.de/IP/elsa](https://gitlab.lrz.de/IP/elsa). It is available under the
Apache 2 open source license.

[[_TOC_]]

## Features

* Multiple optimized forward models for:
  * Attenuation X-ray computed tomography (CT)
  * Phase-Contrast X-ray CT based on grating interferometry
  * Anisotropic Dark-field CT
* Iterative reconstruction algorithms
  * Landweber type algorithms (Landweber, SIRT)
  * Conjugate gradient
  * First-order methods (gradient descent, Nesterov's fast gradient method, optimized gradient method)
  * Proximal gradient methods (proximal gradient descent / ISTA, accelerated gradient descent / FISTA)
  * Alternating Direction Method of Multipliers (ADMM)
* Regularization
  * L2 or Tikhonov based regularization
  * L1 or sparsity inducing regularization
  * Total Variation (TV) regularization
  * Plug-and-Play (PnP) Priors
* Synthetic Phantoms (2D and 3D)
  * Shepp-Logan phantom
  * FORBILD head phantom

For our X-ray CT based imaging modalities, we support arbitrary trajectories.

## Installation and Building

### Requirements

elsa requires a **C++17 compliant compiler**, such as GCC or Clang in recent
versions. Current testing includes Linux-based GNU GCC 11.1.0, and clang 10 to
13. The build process is controlled using CMake, version 3.14 or higher.

The main third party dependencies (Eigen3, spdlog, doctest) are integrated via
[CPM](https://github.com/TheLartians/CPM.cmake).

For CUDA support, you need a CUDA capable graphics card as well as an installation of the CUDA toolkit.
Current testing includes CUDA versions 11.5 to 11.7 combined with GNU GCC 11 as
host compiler.

If you are running an Ubuntu 22.04 based Linux distribution, you can run the
following commands to install the required dependencies for _elsa_:

```bash
apt install git build-essential cmake ninja-build
```

If you plan to use the Python bindings, and/or follow the Python guide in our documentation, you'd
want to install the following packages in an environment:

```bash
apt install python3 python3-pip
pip install numpy matplotlib scipy
```

### Python

If you want to utilize the Python bindings, simply run the following command
from the root directory:

```bash
pip install .
```

This will build **elsa** including the Python bindings. To see if the
installation was successful run:

```bash
python -c "import pyelsa as elsa; print('CUDA enabled') if elsa.cudaProjectorsEnabled() else print('CUDA disabled')"
```
Which will further indicate, if **elsa** is using CUDA.

### C++

Compilation can also be done using CMake. Create a build folder (e.g.
`mkdir build; cd build`) and run the following commands to configure and build
**elsa**:

```bash
cmake ..
make
```

If you want to install, run `make install` after the above commands. This
performs installation to `$DESTDIR/$PREFIX/$elsa-components` (with the defaults
being `DESTDIR=/` and `PREFIX=/usr/local`).

You can change the installation directory prefix path by calling `make
DESTDIR=/some/where`. To change the projects `PREFIX`, you need to tell CMake
about it: `cmake -DCMAKE_INSTALL_PREFIX=/some/path` (by default, it's
`/usr/local`). If you want to build with [Ninja](https://ninja-build.org/)
instead of make, CMake needs to generate Ninja files: `cmake -G Ninja`.

To build and run all tests, run
```bash
make tests
```
from the build directory. To run specific tests, use `make test_SpecificTest`.

You can also use the provided Makefile. This is a convenience wrapper around
CMake to make certain aspects a little easier. So, if in trouble you can always
fall back to the CMake version.

To build **elsa** using the Makefile run:

```bash
make build
```

Calling make will configure the project with certain default configurations and
create a sub-folder structured of the form `build/$BUILD_TYPE/$compiler`.

To run all tests run (from the root directory):

```bash
make tests
```

### Using **elsa** as a library

When using the **elsa** library in your project, we suggest using CMake as the
build system. Once installed, you can configure **elsa** via the `find_package(elsa)`
statement and link your target against elsa with
`target_link_libraries(myTarget elsa::all)`. Alternatively, you can link more
specifically against the required elsa modules, such as
`target_link_libraries(myTarget elsa::core)`.

In your source code, `#include "elsa.h"` to include all of elsa; alternatively,
include only the header files you are actually using to minimize compilation
times. The last step is required, if you are linking against submodules of **elsa**.

### Troubleshooting

Here, we try to gather some common troubleshooting steps,
which might occure when building **elsa**.

If you are facing any issues not listed here, feel free to open an issue, ask
in our Matrix room or contact one of the maintainers.

##### Setting CMake arguments for Python build

When building the Python bindings, you can prefix the call to `pip` with:

```bash
CMAKE_ARGS=<insert-cmake-arguments-or-options>
```
This is especially useful to build our Python bindings in e.g. Debug mode, or
help point CMake to Thrust.

##### Thrust not found

Often, Thrust is not found, if installed standalone or via the CUDA
installation. Specifically, its `thrust-config.cmake` isn't found. To persuade
CMake to locate Thrust, please set
`-DThrust_DIR=/path/where/thrust/cmake/config/is/`. If CUDA is installed, this
directory will usually be `/path/to/cuda/lib64/cmake/thrust`.

## Usage and Documentation

The current documentation of the master branch is available [here](https://ciip.cit.tum.de/elsadocs/).
There are also two guides using the C++ and Python API:
* [C++ guide](https://ciip.cit.tum.de/elsadocs/guides/quickstart-cxx.html)
* [Python guide](https://ciip.cit.tum.de/elsadocs/guides/python_guide/index.html)

Check the example folder for specific cases and more code examples.

Contributing
------------

Do you want to contribute in some way? We appreciate and welcome contributions
each and from everyone. Feel free to join our
[Matrix chat room](https://matrix.to/#/#elsa:in.tum.de) and chat
with us, about areas of interest! Further, see our
[contributing page](https://gitlab.lrz.de/IP/elsa/-/blob/master/CONTRIBUTING.md).

We also have a couple of defined projects, which you can have a look at
[here](https://gitlab.lrz.de/IP/elsa/-/issues/?sort=created_date&state=opened&label_name%5B%5D=student%20project)

Contributors
------------

The **contributors** to elsa are:

* Tobias Lasser
* Matthias Wieczorek
* Jakob Vogel
* David Frank
* Maximilian Hornung
* Nikola Dinev
* Jens Petit
* David Tellenbach
* Jonas Jelten
* Andi Braimllari
* Michael Loipfuehrer
* Jonas Buerger


History
-------

elsa started its life as an internal library at the [Computational Imaging and Inverse Problems](https://ciip.in.tum.de) group at the [Technical University of Munich](https://www.tum.de).
This open-source version is a modernized and cleaned up version of our internal code and will contain most of its functionality, modulo some parts which we unfortunately cannot share (yet).

**Releases:** ([changelog](CHANGELOG.md))

- v0.7: major feature release, e.g. deep learning support (October 27, 2021)
- v0.6: major feature release, e.g. seamless GPU-computing, Python bindings (February 2, 2021)
- v0.5: the "projector" release (September 18, 2019)
- v0.4: first public release (July 19, 2019)

Citation
--------

If you are using elsa in your work, we would like to ask you to cite us:

```txt
@inproceedings{LasserElsa2019,
author = {Tobias Lasser and Maximilian Hornung and David Frank},
title = {{elsa - an elegant framework for tomographic reconstruction}},
volume = {11072},
booktitle = {15th International Meeting on Fully Three-Dimensional Image Reconstruction in Radiology and Nuclear Medicine},
editor = {Samuel Matej and Scott D. Metzler},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {570 -- 573},
keywords = {tomography, tomographic reconstruction, inverse problems, software framework, C++, Python},
year = {2019},
doi = {10.1117/12.2534833},
URL = {https://doi.org/10.1117/12.2534833}
}
```
