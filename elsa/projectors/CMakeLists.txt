cmake_minimum_required(VERSION 3.9)

# list all the headers of the module
set(MODULE_HEADERS
        Geometry.h
        BoundingBox.h
        Intersection.h
        TraverseAABB.h
        TraverseAABBJosephsMethod.h
        BinaryMethod.h
        SiddonsMethod.h
        JosephsMethod.h)

# list all the code files of the module
set(MODULE_SOURCES
        Geometry.cpp
        BoundingBox.cpp
        Intersection.cpp
        TraverseAABB.cpp
        TraverseAABBJosephsMethod.cpp
        BinaryMethod.cpp
        SiddonsMethod.cpp
        JosephsMethod.cpp)

# setup library see elsa/CMakeLists.txt (it's a macro, so we can set variables here)
ELSA_LIBRARY_SETUP(projectors elsa_core elsa_logging)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()