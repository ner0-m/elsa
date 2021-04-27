# set the name of the module
set(ELSA_MODULE_NAME wavelets)
set(ELSA_MODULE_TARGET_NAME elsa_wavelets)
set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)

# list all the headers of the module
set(MODULE_HEADERS
        Wavelet.h
        HaarWavelet.h
        Curvelet.h
        Shearlet.h
        DiscreteShearlet.h
        ConeAdaptedDiscreteShearlet.h
        BootstrappingShearletAndDLModel.h)

# list all the code files of the module
set(MODULE_SOURCES
        Wavelet.cpp
        HaarWavelet.cpp
        Curvelet.cpp
        Shearlet.cpp
        DiscreteShearlet.cpp
        ConeAdaptedDiscreteShearlet.cpp
        BootstrappingShearletAndDLModel.cpp)

# build the module library
add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC elsa_core elsa_logging elsa_proximity_operators elsa_problems elsa_solvers)

# use OpenMP is available
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif ()

target_include_directories(${ELSA_MODULE_TARGET_NAME}
        PUBLIC
        $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        )

# require C++17
target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cxx_std_17)
# set -fPIC
set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# build the tests (if enabled)
if (ELSA_TESTING)
    add_subdirectory(tests)
endif (ELSA_TESTING)

if (ELSA_BUILD_PYTHON_BINDINGS)
    GENERATE_BINDINGS(${ELSA_MODULE_TARGET_NAME}
            bind_${ELSA_MODULE_NAME}.cpp
            ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.hpp
            ${MODULE_SOURCES})
endif ()

# register the module
registerComponent(${ELSA_MODULE_NAME})

# install the module
InstallElsaModule(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})
