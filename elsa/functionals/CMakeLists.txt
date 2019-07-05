cmake_minimum_required(VERSION 3.9)

# set the name of the module
set(ELSA_MODULE_NAME functionals)
set(ELSA_MODULE_TARGET_NAME elsa_functionals)
set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)


# list all the headers of the module
set(MODULE_HEADERS
        Residual.h
        LinearResidual.h
        Functional.h
        L1Norm.h
        L2NormPow2.h
        WeightedL2NormPow2.h
        LInfNorm.h
        Huber.h
        PseudoHuber.h
        Quadric.h
        EmissionLogLikelihood.h
        TransmissionLogLikelihood.h)

# list all the code files of the module
set(MODULE_SOURCES
        Residual.cpp
        LinearResidual.cpp
        Functional.cpp
        L1Norm.cpp
        L2NormPow2.cpp
        WeightedL2NormPow2.cpp
        LInfNorm.cpp
        Huber.cpp
        PseudoHuber.cpp
        Quadric.cpp
        EmissionLogLikelihood.cpp
        TransmissionLogLikelihood.cpp)


# build the module library
add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

target_link_libraries(${ELSA_MODULE_TARGET_NAME} elsa_core elsa_logging elsa_operators)

target_include_directories(${ELSA_MODULE_TARGET_NAME}
        PUBLIC
        $<INSTALL_INTERFACE:include/${ELSA_MODULE_NAME}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        )


# require C++17
target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cxx_std_17)


# build the tests (if enabled)
if(ELSA_TESTING)
    add_subdirectory(tests)
endif(ELSA_TESTING)


# register the module
registerComponent(${ELSA_MODULE_NAME})

# install the module
InstallElsaModule(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})
