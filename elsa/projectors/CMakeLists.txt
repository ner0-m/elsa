# list all the headers of the module
set(MODULE_HEADERS
    BoundingBox.h
    Intersection.h
    TraverseAABB.h
    TraverseAABBJosephsMethod.h
    SliceTraversal.h
    BinaryMethod.h
    SiddonsMethod.h
    JosephsMethod.h
    SubsetSampler.h
)

# list all the code files of the module
set(MODULE_SOURCES
    BoundingBox.cpp
    Intersection.cpp
    TraverseAABB.cpp
    TraverseAABBJosephsMethod.cpp
    SliceTraversal.cpp
    BinaryMethod.cpp
    SiddonsMethod.cpp
    JosephsMethod.cpp
    SubsetSampler.cpp
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

if(ELSA_BUILD_PYTHON_BINDINGS)
    GENERATE_BINDINGS(
        ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
        ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${MODULE_SOURCES}
    )
endif()
 
write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging)
