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

target_compile_options(elsa_quickvec PUBLIC --extended-lambda --expt-relaxed-constexpr -Xcudafe "--diag_suppress=implicit_return_from_non_void_function")

write_module_config(${ELSA_MODULE_NAME})
