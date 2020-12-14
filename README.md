elsa - an elegant framework for tomographic reconstruction
==========================================================

**elsa** is a modern, flexible C++ library intended for use in tomographic reconstruction. Currently, operators are implemented for X-ray Computed Tomography. Other imaging modalities can be supported by implementing appropriate operators.

CI Status (master)
---------------
![Pipeline status (master)](https://gitlab.lrz.de/IP/elsa/badges/master/pipeline.svg)
![Coverage status (master)](https://gitlab.lrz.de/IP/elsa/badges/master/coverage.svg)

Documentation
-------------

The documentation is available [here](https://ciip.in.tum.de/elsadocs/).


Requirements
------------

elsa requires a **C++17 compliant compiler**, such as GCC, Clang or Microsoft Visual Studio in recent versions.
Current testing includes gcc7, gcc9, and clang8.

The main third party dependencies (Eigen3, spdlog, Catch2) are integrated via [CPM](https://github.com/TheLartians/CPM.cmake).

For CUDA support, you need a CUDA capable graphics card as well as an installation of the CUDA toolkit.
Current testing includes CUDA 9.2 combined with gcc7.

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

You can provide `-DCMAKE_INSTALL_PREFIX=folder` during the cmake step to select an installation destination other than the default (`/usr/local` on Unix).

You can build and run the elsa unit tests by running (in the build folder):
```
make tests
```

Building against the elsa library
---------------------------------

When using the elsa library in your project, we suggest using CMake as the build system.
Then you can configure elsa via the `find_package(elsa)` statement and link your target against elsa with `target_link_libraries(myTarget elsa::all)`.
Alternatively, you can link more specifically only against the required elsa modules, such as `target_link_libraries(myTarget elsa::core)`.
In your source code, `#include "elsa.h"` to include all of elsa; alternatively, include only the header files you are actually using to minimize compilation times.

As elsa depends on Eigen3 (version 3.3 or newer) and spdlog (version 1.0 or newer), you will need to have these packages installed on your system, and you have to point CMake to those installations.
When using CUDA, your CMake should be version 3.14 or newer.

Contributing
------------
To get involved, please see our [contributing page](CONTRIBUTING.md).

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

History
-------

elsa started its life as an internal library at the [Computational Imaging and Inverse Problems](https://ciip.in.tum.de) group at the [Technical University in Munich](https://www.tum.de).
This open-source version is a modernized and cleaned up version of our internal code and will contain most of its functionality, modulo some parts which we unfortunately cannot share (yet).

**Releases:**

- v0.5: the "projector" release (September 18, 2019)
- v0.4: first public release (July 19, 2019)
