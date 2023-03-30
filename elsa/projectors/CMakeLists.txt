# list all the headers of the module
set(MODULE_HEADERS
    BoundingBox.h
    Intersection.h
    TraverseAABB.h
    DrivingDirectionTraversal.h
    SliceTraversal.h
    LutProjector.h
    VoxelProjector.h
    BinaryMethod.h
    SiddonsMethod.h
    JosephsMethod.h
    SubsetSampler.h
    Blobs.h
    BSplines.h
    Luts.hpp
    XrayProjector.h
)

# list all the code files of the module
set(MODULE_SOURCES
    BoundingBox.cpp
    Intersection.cpp
    TraverseAABB.cpp
    DrivingDirectionTraversal.cpp
    SliceTraversal.cpp
    LutProjector.cpp
    VoxelProjector.cpp
    BinaryMethod.cpp
    SiddonsMethod.cpp
    JosephsMethod.cpp
    SubsetSampler.cpp
    Blobs.cpp
    BSplines.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging)
list(APPEND MODULE_PRIVATE_DEPS)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    list(APPEND MODULE_PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif()

ADD_ELSA_MODULE(
    projectors "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging)
