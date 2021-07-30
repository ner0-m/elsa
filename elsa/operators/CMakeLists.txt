# list all the headers of the module
set(MODULE_HEADERS
    Identity.h
    Scaling.h
    FiniteDifferences.h
    FourierTransform.h
    BlockLinearOperator.h
    Dictionary.h)

# list all the code files of the module
set(MODULE_SOURCES
    Identity.cpp
    Scaling.cpp
    FiniteDifferences.cpp
    FourierTransform.cpp
    BlockLinearOperator.cpp
    Dictionary.cpp)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging)
list(APPEND MODULE_PRIVATE_DEPS)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    list(APPEND MODULE_PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif()

ADD_ELSA_MODULE(
    operators "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

# link Quickvec for DataHandlerGPU
# Also add the necessary compile definition for quickvec
if(${ELSA_BUILD_WITH_QUICKVEC})
    set_source_files_properties(${MODULE_SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(${ELSA_MODULE_TARGET_NAME} PUBLIC ELSA_ENABLE_CUDA_VECTOR)
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC "Quickvec::quickvec")
    target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cuda_std_17)
endif()

if(ELSA_BUILD_PYTHON_BINDINGS)
    GENERATE_BINDINGS(
        ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
        ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${MODULE_SOURCES}
    )
endif()
 
write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging)
