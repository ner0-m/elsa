cmake_minimum_required(VERSION 3.9)

# list all the headers of the module
set(MODULE_HEADERS
        RegularizationTerm.h
        Problem.h
        WLSProblem.h
        QuadricProblem.h)

# list all the code files of the module
set(MODULE_SOURCES
        RegularizationTerm.cpp
        Problem.cpp
        WLSProblem.cpp
        QuadricProblem.cpp)

# setup library see elsa/CMakeLists.txt (it's a macro, so we can set variables here)
ELSA_LIBRARY_SETUP(problems elsa_core elsa_logging elsa_operators elsa_functionals)