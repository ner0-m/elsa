cmake_minimum_required(VERSION 3.9)

# list all the headers of the module
set(MODULE_HEADERS
        EDFHandler.h
        MHDHandler.h
        ioUtils.h)

# list all the code files of the module
set(MODULE_SOURCES
        EDFHandler.cpp
        MHDHandler.cpp
        ioUtils.cpp)

# setup library see elsa/CMakeLists.txt (it's a macro, so we can set variables here)
ELSA_LIBRARY_SETUP(io elsa_core elsa_logging)