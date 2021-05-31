elsa - an elegant framework for tomographic reconstruction
==========================================================

**elsa** is a modern, flexible C++ library intended for use in tomographic reconstruction.
Using concepts such as data containers, operators, and functionals, inverse problems can be modelled and then solved.
**elsa** supports any imaging modality in general, but currently only implements forward models for X-ray Computed Tomography.
Seamless GPU computing based on CUDA is supported, along with Python bindings for ease of use.

Continuous Integration status (master)
---------------
![Pipeline status (master)](https://gitlab.lrz.de/IP/elsa/badges/master/pipeline.svg)
![Coverage status (master)](https://gitlab.lrz.de/IP/elsa/badges/master/coverage.svg)

Documentation
-------------

Have a look at the [quickstart guide](https://ciip.in.tum.de/elsadocs/guides/quickstart-cxx.html), an [Alternating
Direction Method of Multipliers solver guide](), or check out the current
documentation [here](https://ciip.in.tum.de/elsadocs/).

Requirements
------------

elsa requires a **C++17 compliant compiler**, such as GCC or Clang in recent versions.
Current testing includes Linux-based gcc 9, gcc 10, clang 9, and clang 10.
The build process is controlled using CMake, version 3.14 or higher.

The main third party dependencies (Eigen3, spdlog, Catch2) are integrated via [CPM](https://github.com/TheLartians/CPM.cmake).

For CUDA support, you need a CUDA capable graphics card as well as an installation of the CUDA toolkit.
Current testing includes CUDA 10.2 combined with gcc 8 or clang 8.

Compiling
---------

Once you have cloned the git repository, compilation can be done by following these steps:

- go to the elsa folder and create a build folder (e.g. `mkdir build; cd build`)
- run the following commands in the build folder:

```
cmake ..
make
make install
```

You can provide `-DCMAKE_INSTALL_PREFIX=folder` during the cmake step to select an installation destination other than the default (`/usr/local` on Unix-like systems).

You can build and run the elsa unit tests by running (in the build folder):
```
make tests
```

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

As elsa depends on Eigen3 (version 3.3.9 or newer) and spdlog (version 1.7 or newer), you will need to have these packages installed on your system, and you have to point CMake to those installations.

Contributing
------------
To get involved, please see our [contributing page](https://gitlab.lrz.de/IP/elsa/-/blob/master/CONTRIBUTING.md).

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


History
-------

elsa started its life as an internal library at the [Computational Imaging and Inverse Problems](https://ciip.in.tum.de) group at the [Technical University of Munich](https://www.tum.de).
This open-source version is a modernized and cleaned up version of our internal code and will contain most of its functionality, modulo some parts which we unfortunately cannot share (yet).

**Releases:** ([changelog](CHANGELOG.md))

- v0.6: major feature release, e.g. seamless GPU-computing, Python bindings (February 2, 2021)
- v0.5: the "projector" release (September 18, 2019)
- v0.4: first public release (July 19, 2019)
