|Pipeline status|

.. |Pipeline status| image:: https://gitlab.lrz.de/IP/elsa/badges/master/pipeline.svg
   :target: https://gitlab.lrz.de/IP/elsa/commits/master

elsa - an elegant framework for tomographic reconstruction
==========================================================

**elsa** is a modern, flexible C++ library intended for use in tomographic reconstruction.
Currently, operators are implemented for X-ray Computed Tomography. 
Other imaging modalities can be supported by implementing appropriate operators.

Documentation
-------------

The documentation is available `here <https://ip.campar.in.tum.de/elsadocs/>`_.


Requirements
------------

elsa requires a *C++17 compliant compiler*, such as GCC, Clang or Microsoft Visual Studio in recent versions.
Current testing includes gcc7, gcc9, and clang8.

The main third party dependencies (Eigen3, spdlog, Catch2) are integrated via git submodules.


Compiling
---------

Once you have cloned the git repository, compilation can be done by following these steps:

   - go to the elsa folder and create a build folder (e.g. mkdir build; cd build)
   - run the following commands in the build folder:

      - cmake ..
      - make
      - make install

You can provide *-DCMAKE_INSTALL_PREFIX=folder* during the cmake step to select an installation destination other than the default (/usr/local on Unix).

You can run the elsa unit tests by running (in the build folder):

   - ctest


Building against the elsa library
---------------------------------

When using the elsa library in your project, we advise using CMake as the build system. You can then include elsa via the *find_package(elsa)* statement and linking your target against the corresponding elsa modules, e.g. *target_link_libraries(myTarget elsa::core)*.

As elsa depends on Eigen3 and spdlog, you will need to have these packages installed on your system, and you have to point CMake to those installations.


Contributors
------------

The *contributors* to elsa are:

   - Tobias Lasser
   - Matthias Wieczorek
   - Jakob Vogel
   - David Frank
   - Maximilian Hornung
   - Nikola Dinev

History
-------

elsa started its life as an internal library at the `Inverse Problems in Tomography <https://ip.campar.in.tum.de>`_ group at the `Technical University in Munich <https://www.tum.de>`_.
This open-source version is a modernized and cleaned up version of our internal code and will contain most of its functionality, modulo some parts which we unfortunately cannot share (yet).

*Releases:*

   - v0.4: first public release (July 19, 2019)
