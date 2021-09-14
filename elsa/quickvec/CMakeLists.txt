# ------------ setup VectorCUDA --------
# ------------
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
find_package(CUDA 11.0 REQUIRED)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)


    message(STATUS "CUDA available - requirements fullfilled.")

    include(ComputeCapabilityCUDA)

    set_cuda_arch_type(${ELSA_CUDA_ARCH_TYPE})

    foreach(TARGET_GPU ${TARGET_GPUS})
        string(REPLACE "." "" TARGET_GPU ${TARGET_GPU})
        list(APPEND CMAKE_CUDA_ARCHITECTURES ${TARGET_GPU})
    endforeach()

    message(STATUS "GPU architectures ${CMAKE_CUDA_ARCHITECTURES}!")
else()
    message(FATAL_ERROR "CUDA compiler not found in appropiate version!")
endif()

set(MODULE_HEADERS
    Helpers.cuh
    Vector.cuh
    Expression.cuh
    SharedPointer.cuh)

set(MODULE_SOURCES
    Vector.cu
    SharedPointer.cu)

list(APPEND MODULE_PUBLIC_DEPS Eigen3::Eigen ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

ADD_ELSA_MODULE(
    quickvec "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

target_compile_options(elsa_quickvec PUBLIC --extended-lambda --expt-relaxed-constexpr -Xcudafe "--diag_suppress=implicit_return_from_non_void_function --diag_suppress=unrecognized_pragma")

write_module_config(${ELSA_MODULE_NAME})
