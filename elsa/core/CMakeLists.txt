cmake_minimum_required(VERSION 3.9)

# list all the headers of the module
set(MODULE_HEADERS
        elsaDefines.h
        Cloneable.h
        DataDescriptor.h
        BlockDescriptor.h
        DataContainer.h
        DataContainerIterator.h
        DataHandler.h
        DataHandlerCPU.h
        DataHandlerMapCPU.h
        LinearOperator.h)

# list all the code files of the module
set(MODULE_SOURCES
        DataDescriptor.cpp
        BlockDescriptor.cpp
        DataContainer.cpp
        DataHandlerCPU.cpp
        DataHandlerMapCPU.cpp
        LinearOperator.cpp)

ELSA_LIBRARY_SETUP(core)

target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC Eigen3::Eigen)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()