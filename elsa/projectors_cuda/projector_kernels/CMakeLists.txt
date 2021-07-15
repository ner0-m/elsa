include(ComputeCapabilityCUDA)

# make sure host compiler used by NVCC is the one used for the rest of the project
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

# Append compute capabilities to CMAKE_CUDA_ARCHITECTURE 
set_cuda_arch_type(${ELSA_CUDA_ARCH_TYPE})

# fmad can reduce accuracy, but only gives a negligible boost to speed
# use expt-relaxed-constexpr flag to suppress warnings caused by including the Eigen header
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --expt-relaxed-constexpr")

if(ELSA_MASTER_PROJECT)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wreorder -Xptxas=-warn-spills,-warn-lmem-usage")
endif()

# set the name of the module
set(ELSA_MODULE_NAME projector_kernels)
set(ELSA_MODULE_TARGET_NAME elsa_${ELSA_MODULE_NAME})
set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)

# list all the headers of the module
set(MODULE_HEADERS
        TraverseSiddonsCUDA.cuh
        TraverseJosephsCUDA.cuh)

# list all the code files of the module
set(MODULE_SOURCES
        TraverseSiddonsCUDA.cu
        TraverseJosephsCUDA.cu)

# build the module library
add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

#CUDA doesn't support C++17, require C++14
set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES CUDA_STANDARD 14 POSITION_INDEPENDENT_CODE ON)

# CUDA doesn't support linking against a library built under the C++17 standard,
# we only need the elsa.h header -> manually specify include directories
target_include_directories(${ELSA_MODULE_TARGET_NAME} PUBLIC 
        $<TARGET_PROPERTY:elsa_core,INTERFACE_INCLUDE_DIRECTORIES>
        $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC cudart)

# register the module
REGISTER_MODULE(${ELSA_MODULE_NAME})

# install the kernels as a module
install_elsa_module(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})
 
write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES)
