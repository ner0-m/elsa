cmake_minimum_required(VERSION 3.9)

# set the name of the module
set(ELSA_MODULE_NAME core)
set(ELSA_MODULE_TARGET_NAME elsa_core)
set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)


# list all the headers of the module
set(MODULE_HEADERS
        elsaDefines.h
        Cloneable.h
        DataDescriptor.h
        BlockDescriptor.h
        IdenticalBlocksDescriptor.h
        PartitionDescriptor.h
        RandomBlocksDescriptor.h
        DataContainer.h
        DataContainerIterator.h
        DataHandler.h
        DataHandlerCPU.h
        DataHandlerMapCPU.h
        LinearOperator.h
        Expression.h
        ExpressionPredicates.h)

# list all the code files of the module
set(MODULE_SOURCES
        DataDescriptor.cpp
        RandomBlocksDescriptor.cpp
        IdenticalBlocksDescriptor.cpp
        PartitionDescriptor.cpp
        DataContainer.cpp
        DataHandlerCPU.cpp
        DataHandlerMapCPU.cpp
        LinearOperator.cpp)


# build the module library
add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC Eigen3::Eigen)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()


target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC Eigen3::Eigen)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()


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


# register the module
registerComponent(${ELSA_MODULE_NAME})

# install the module
InstallElsaModule(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})
