cmake_minimum_required(VERSION 3.9)

# list all the headers of the module
set(MODULE_HEADERS
        Identity.h
        Scaling.h
        FiniteDifferences.h)

# list all the code files of the module
set(MODULE_SOURCES
        Identity.cpp
        Scaling.cpp
        FiniteDifferences.cpp)

# setup library see elsa/CMakeLists.txt (it's a macro, so we can set variables here)
ELSA_LIBRARY_SETUP(operators elsa_core elsa_logging)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()