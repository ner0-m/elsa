Change Log
==========

All notable changes are documented in this file.

v0.7 (October 27, 2021)
-----------------------
- add deep learning module, with backends for cuDNN and OneDNN
- add initial support for dictionary learning (representation problem using OMP)
- add ADMM solver
- add preconditioning support to FGM, OGM, SQS
- add ordered subset support for SQS
- upgrades to DataContainers (e.g. reductions)
- replace Catch2 with doctest
- many improvements to the CMake setup
- add support for CMake presets
- add linting for CMake files
- use docker registry for CI images
- updates to documentation, add some guides
- improvements on unit tests
- various code clean-ups
- various bugfixes


v0.6 (February 2, 2021)
-----------------------
- switch to CPM for third party dependencies (away from git submodules)
- add code coverage with lcov
- update CI pipelines to test more, including installation and building examples
- update and modularize docker image generation for CI servers
- add sanitizers and clang-format/clang-tidy to the CI pipeline
- add contributing information file
- add elsa::all CMake convenience target and elsa.h convenience header
- iterator support for DataContainer
- add VolumeDescriptor, DetectorDescriptor, various BlockDescriptors for DataContainer
- add seamless GPU computing via QuickVec and DataHandlerGPU, with expression templates 
- add QuadricProblem, TikhonovProblem, LASSOProblem
- add CG, FGM, OGM, ISTA, FISTA solvers
- add proximity operators (so far: hard- and soft-thresholding)
- add PGM handler for 2D image output
- add automatically generated Python bindings (building on libclang-cpp)
- add benchmarks
- various bugfixes


v0.5 (September 18, 2019)
-------------------------

- added doxygen/Sphinx based documentation
- added CPU projectors using Siddon's and Joseph's method
- enabled OpenMP for operators and projectors
- added GPU projectors using Siddon's and Joseph's method using CUDA
- added simple example programs in example/ folder
- added variadic indexing to DataContainer


v0.4 (July 19, 2019)
--------------------

- first public release