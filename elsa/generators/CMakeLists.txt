cmake_minimum_required(VERSION 3.9)

# set the name of the module
set(ELSA_MODULE_NAME generators)
set(ELSA_MODULE_TARGET_NAME elsa_generators)
set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)


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


# build the module library
add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC elsa_core elsa_logging elsa_projectors)

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
if(ELSA_TESTING)
    add_subdirectory(tests)
endif(ELSA_TESTING)

if (ELSA_BUILD_PYTHON_BINDINGS)
        GENERATE_BINDINGS(${ELSA_MODULE_TARGET_NAME}
                ${CMAKE_CURRENT_BINARY_DIR}/bind_${ELSA_MODULE_NAME}.cpp
                ""
                ${MODULE_SOURCES} CircleTrajectoryGenerator.h)
endif()

# register the module
registerComponent(${ELSA_MODULE_NAME})

# install the module
InstallElsaModule(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})
