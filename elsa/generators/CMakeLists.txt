cmake_minimum_required(VERSION 3.9)

# list all the headers of the module
set(MODULE_HEADERS
        EllipseGenerator.h
        PhantomGenerator.h
        CircleTrajectoryGenerator.h)

# list all the code files of the module
set(MODULE_SOURCES
        EllipseGenerator.cpp
        PhantomGenerator.cpp
        CircleTrajectoryGenerator.cpp)

# setup library see elsa/CMakeLists.txt (it's a macro, so we can set variables here)
ELSA_LIBRARY_SETUP(generators elsa_core elsa_logging elsa_projectors)
