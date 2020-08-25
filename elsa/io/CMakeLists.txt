cmake_minimum_required(VERSION 3.9)

# set the name of the module
set(ELSA_MODULE_NAME io)
set(ELSA_MODULE_TARGET_NAME elsa_io)
set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)


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


# build the module library
add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC elsa_core elsa_logging)

target_include_directories(${ELSA_MODULE_TARGET_NAME}
        PUBLIC
        $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        )


# require C++17
target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cxx_std_17)
# set -fPIC
set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (ELSA_BUILD_PYTHON_BINDINGS)
        set(BINDINGS_SOURCES ${MODULE_SOURCES})

        list(FILTER BINDINGS_SOURCES EXCLUDE REGEX ioUtils.cpp)
        
        GENERATE_BINDINGS(${ELSA_MODULE_TARGET_NAME}
                bind_${ELSA_MODULE_NAME}.cpp
                ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.hpp
                ${BINDINGS_SOURCES})
endif()


# build the tests (if enabled)
if(ELSA_TESTING)
    add_subdirectory(tests)
endif(ELSA_TESTING)


# register the module
registerComponent(${ELSA_MODULE_NAME})

# install the module
InstallElsaModule(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})