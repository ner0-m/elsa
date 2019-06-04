# elsa - an elegant framework for tomographic reconstruction

*elsa* is a modern, flexible C++ library intended for use in tomographic reconstruction.
Currently, operators are implemented for X-ray Computed Tomography. 
Other imaging modalities can be supported by implementing appropriate operators.


## Pre-release status

elsa is currently in *pre-release status* and not yet fully functional.
However, due to the Fully3D conference on June 2-6, 2019, a subset of the functionality is already available for your perusal.
The missing modules will be brought online shortly.


# Requirements

elsa requires a *C++17 compliant compiler*, such as GCC, Clang or Microsoft Visual Studio in recent versions.
Current testing includes gcc7, gcc9, and clang8.

The main third party dependencies (Eigen3, spdlog, Catch2) are integrated via git submodules.


# Compiling

Once you have cloned the git repository, compilation can be done by following these steps:

   * go to the elsa folder and create a build folder (e.g. mkdir build; cd build)
   * run the following commands in the build folder:
      * cmake ..
      * make
      * make install

You can provide *-DCMAKE_INSTALL_PREFIX=folder* during the cmake step to select an installation destination other than the default (/usr/local on Unix).

You can run the elsa unit tests by running *ctest*.


# Building against the elsa library

When using the elsa library in your project, we advise using CMake as the build system. You can then include elsa via the *find_package(elsa)* statement and linking your target against the corresponding elsa modules, e.g. *target_link_libraries(myTarget elsa::core)*.

As elsa depends on Eigen3 and spdlog, you will need to have these packages installed on your system, and you have to point CMake to those installations.


# History and contributors

elsa started its life as an internal library at the [Inverse Problems in Tomography](https://ip.campar.in.tum.de) group at the [Technical University in Munich](https://www.tum.de).
This open-source version is a modernized and cleaned up version of our internal code and will contain most of its functionality, modulo some parts which we unfortunately cannot share (yet).

The *contributors* to elsa are:

   * Matthias Wieczorek 
   * Tobias Lasser
   * David Frank
   * Maximilian Hornung
   * Nikola Dinev


