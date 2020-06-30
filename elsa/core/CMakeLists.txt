cmake_minimum_required(VERSION 3.9)

# set the name of the module
set(ELSA_MODULE_NAME core)
set(ELSA_MODULE_TARGET_NAME elsa_core)
set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)


# list all the headers of the module
set(MODULE_HEADERS
        elsaDefines.h
        Cloneable.h
        Descriptors/DataDescriptor.h
        Descriptors/DescriptorUtils.h
        Descriptors/VolumeDescriptor.h
        Descriptors/DetectorDescriptor.h
        Descriptors/PlanarDetectorDescriptor.h
        Descriptors/BlockDescriptor.h
        Descriptors/IdenticalBlocksDescriptor.h
        Descriptors/PartitionDescriptor.h
        Descriptors/RandomBlocksDescriptor.h
        DataContainer.h
        DataContainerIterator.h
        Handlers/DataHandler.h
        Handlers/DataHandlerCPU.h
        Handlers/DataHandlerMapCPU.h
        LinearOperator.h
        Expression.h
        ExpressionPredicates.h
        Geometry.h
        StrongTypes.h)

# list all the code files of the module
set(MODULE_SOURCES
        Descriptors/DataDescriptor.cpp
        Descriptors/VolumeDescriptor.cpp
        Descriptors/PlanarDetectorDescriptor.cpp
        Descriptors/RandomBlocksDescriptor.cpp
        Descriptors/DescriptorUtils.cpp
        Descriptors/IdenticalBlocksDescriptor.cpp
        Descriptors/DetectorDescriptor.cpp
        Descriptors/PartitionDescriptor.cpp
        DataContainer.cpp
        Handlers/DataHandlerCPU.cpp
        Handlers/DataHandlerMapCPU.cpp
        LinearOperator.cpp
        Geometry.cpp
        StrongTypes.cpp)

if(${ELSA_CUDA_VECTOR})
    set(MODULE_HEADERS
        ${MODULE_HEADERS}
        Handlers/DataHandlerGPU.h
        Handlers/DataHandlerMapGPU.h)

    set(MODULE_SOURCES
        ${MODULE_SOURCES}
        Handlers/DataHandlerGPU.cpp
        Handlers/DataHandlerMapGPU.cpp)
endif()

# build the module library
add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC Eigen3::Eigen)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# link Quickvec for DataHandlerGPU
if(ELSA_CUDA_VECTOR AND Quickvec_FOUND)
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC Quickvec::quickvec)
    target_compile_definitions(${ELSA_MODULE_TARGET_NAME} PUBLIC ELSA_ENABLE_CUDA_VECTOR)
endif()

target_include_directories(${ELSA_MODULE_TARGET_NAME}
        PUBLIC
        $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}>
        $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}/Descriptors>
        $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}/Handlers>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Descriptors>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Handlers>
        )


# require C++17
target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cxx_std_17)
# set -fPIC
set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (ELSA_BUILD_PYTHON_BINDINGS)
    set(BINDINGS_SOURCES ${MODULE_SOURCES}
        Descriptors/DataDescriptor.h
        Descriptors/DescriptorUtils.h
        Descriptors/VolumeDescriptor.h
        Descriptors/BlockDescriptor.h
        Descriptors/IdenticalBlocksDescriptor.h
        Descriptors/PartitionDescriptor.h
        Descriptors/RandomBlocksDescriptor.h
        Descriptors/DetectorDescriptor.h
        Descriptors/PlanarDetectorDescriptor.h
        Geometry.h)
        
    list(FILTER BINDINGS_SOURCES EXCLUDE REGEX DataHandler.*.cpp)
    list(FILTER BINDINGS_SOURCES EXCLUDE REGEX .*Descriptor.*.cpp)
    list(FILTER BINDINGS_SOURCES EXCLUDE REGEX Geometry.cpp)
    list(FILTER BINDINGS_SOURCES EXCLUDE REGEX StrongTypes.cpp)
    

    GENERATE_BINDINGS(${ELSA_MODULE_TARGET_NAME}
            ${CMAKE_CURRENT_BINARY_DIR}/bind_${ELSA_MODULE_NAME}.cpp
            bindings_hints.hpp
            ${BINDINGS_SOURCES})
endif()

# build the tests (if enabled)
if(ELSA_TESTING)
    add_subdirectory(tests)
endif(ELSA_TESTING)


# register the module
registerComponent(${ELSA_MODULE_NAME})

# install the module
InstallElsaModuleDir(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/")
