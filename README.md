elsa - an elegant framework for tomographic reconstruction
==========================================================

**elsa** is a modern, flexible C++ library intended for use in tomographic reconstruction.
Using concepts such as data containers, operators, and functionals, inverse problems can be modelled and then solved.
**elsa** supports any imaging modality in general, but currently only implements forward models for X-ray Computed Tomography.
Seamless GPU computing based on CUDA is supported, along with Python bindings for ease of use.
The source code can be found [here](https://gitlab.lrz.de/IP/elsa).

The framework is mostly developed by the Computational Imaging and Inverse Problems
(CIIP) group at the Technical University of Munich. For more info about our research
checkout our [website](https://ciip.in.tum.de/).

Continuous Integration status (master)
---------------

![Pipeline status (master)](https://gitlab.lrz.de/IP/elsa/badges/master/pipeline.svg)
![Coverage status (master)](https://gitlab.lrz.de/IP/elsa/badges/master/coverage.svg)

Documentation
-------------

The current documentation of the master branch is available [here](https://ciip.in.tum.de/elsadocs/).
There is also
*  a [quickstart guide](https://ciip.in.tum.de/elsadocs/guides/quickstart-cxx.html)
*  a [guide for Python bindings](https://ciip.in.tum.de/elsadocs/guides/python_guide/index.html)
*  a [tutorial for choosing solvers](https://ciip.in.tum.de/elsadocs/modules/solvers/choosing_a_solver.html)
*  a [tutorial on the Alternating Direction Method of Multipliers solver](https://ciip.in.tum.de/elsadocs/guides/admm-cxx.html)

Requirements
------------

elsa requires a **C++17 compliant compiler**, such as GCC or Clang in recent versions.
Current testing includes Linux-based gcc 9, gcc 10, clang 9, and clang 10.
The build process is controlled using CMake, version 3.14 or higher.

The main third party dependencies (Eigen3, spdlog, doctest) are integrated via [CPM](https://github.com/TheLartians/CPM.cmake).

For CUDA support, you need a CUDA capable graphics card as well as an installation of the CUDA toolkit.
Current testing includes CUDA 10.2 combined with gcc 8 or clang 8.

If you are running an Ubuntu 22.04 based Linux distribution, you can run the following commands to
install the minimum required dependencies for _elsa_:

```bash
apt install git build-essential cmake ninja-build
```

If you plan to use the Python bindings, and/or follow the Python guide in our documentation, you'd
want to install the following packages in an environment:

```bash
apt install python3 python3-pip
pip install numpy matplotlib scipy
```

Compiling
---------

Once you have cloned the git repository, compilation can be done by simply by running

```
make build
```

The build CMake-based but a Makefile is provided as a convenience.

Calling make will configure the project with certain default configurations and create a sub-folder structured of the
form `build/$BUILD_TYPE/$compiler`.

Currently, if you want to change the install prefix, you have to directly call CMake. Provide `-DCMAKE_INSTALL_PREFIX=folder` during the CMake
step to select an installation destination other than the default (`/usr/local` on Unix-like systems).

To run all tests just run (from the root directory):

```
make tests
```

Once configuration was run once, other interesting targets for developers could be:

* test <test-name>
* watch <test-name>

You might need to install [fzf](https://github.com/junegunn/fzf), [chromaterm](https://github.com/hSaria/ChromaTerm),
[ag](https://github.com/ggreer/the_silver_searcher) and/or [entr](http://eradman.com/entrproject/) for the best
experience. ag and entr are necessary for the watch command. If you have fzf installed, you can also use partial test names and
you can select one interactively.

Other build options you can pass: `USE_CUDA`, `USE_DNNL`, and `GENERATE_PYBINDS`. You can pass either `y` or `n` to any of these.

Compilation can also be done using plain CMake, without the Makefile. For create a build folder (e.g. `mkdir build; cd build`)
and run the following commands:

```bash
cmake ..
make
make install
```

You can provide the usual CMake options with a prefix of `-D` (e.g. `-DCMAKE_INSTALL_PREFIX=path/to/install/dir`)
or use [ninja](https://ninja-build.org/) to build instead of make by appending `-G Ninja` to the CMake call.

We also provide a `CMakePresets.json` to support [CMake's presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html).
You can use them the following way from the root of the repository:

```
cmake --preset=<name>
```

A couple of useful presets provided here are: `default-gcc`, `default-clang`, `default-clang-libcxx` and
`default-coverage`. For more configurations such as configurations with sanitizers check
the `CMakePresets.json` file. The preset names are rather long, that way it's
easy to overwrite them with shorter names in your personal `CMakeUserPresets.json`.

Building against the elsa library
---------------------------------

When using the elsa library in your project, we suggest using CMake as the build system.
Then you can configure elsa via the `find_package(elsa)` statement and link your target against elsa with `target_link_libraries(myTarget elsa::all)`.
Alternatively, you can link more specifically only against the required elsa modules, such as `target_link_libraries(myTarget elsa::core)`.
In your source code, `#include "elsa.h"` to include all of elsa; alternatively, include only the header files you are actually using to minimize compilation times.

Contributing
------------

Do you want to contribute in some way? We appreciate and welcome contributions each and from
everyone. Feel free to join our [Matrix chat room](https://matrix.to/#/#elsa:in.tum.de) and chat
with us, about areas of interest! Further, see our [contributing
page](https://gitlab.lrz.de/IP/elsa/-/blob/master/CONTRIBUTING.md).

We also have a couple of defined projects, which you can have a look at
[here](https://gitlab.lrz.de/IP/elsa/-/issues/?sort=created_date&state=opened&label_name%5B%5D=student%20project)

Contributors
------------

The **contributors** to elsa are:

- Tobias Lasser
- Matthias Wieczorek
- Jakob Vogel
- David Frank
- Maximilian Hornung
- Nikola Dinev
- Jens Petit
- David Tellenbach
- Jonas Jelten
- Andi Braimllari
- Michael Loipfuehrer
- Jonas Buerger


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

If you are using the deep learning module of elsa, we would like to ask you to cite us as well:
```txt
@inproceedings{TellenbachElsa2020,
author = {David Tellenbach and Tobias Lasser},
title = {{elsa - an elegant framework for precision learning in tomographic reconstruction}},
booktitle = {6th International Conference on Image Formation in X-ray Computed Tomography},
venue = {Regensburg, Germany},
month = {August},
year = {2020},
```
