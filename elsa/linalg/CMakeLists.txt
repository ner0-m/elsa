# list all the headers of the module
set(MODULE_HEADERS Matrix.h Vector.h RowVector.h ColumnVector.h)

# list all the code files of the module
set(MODULE_SOURCES Matrix.cpp Vector.cpp RowVector.cpp ColumnVector.cpp)

list(APPEND MODULE_PUBLIC_DEPS elsa::Thrust elsa_config elsa_storage "Eigen3::Eigen")
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    linalg "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

# If CUDA is enabled, all of the reductions should be compiled with CUDA not the host compiler
if(ELSA_CUDA_ENABLED)
    set_target_properties(elsa_linalg PROPERTIES LINKER_LANGUAGE CUDA)

    set_source_files_properties(Matrix.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(Vector.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(RowVector.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(ColumnVector.cpp PROPERTIES LANGUAGE CUDA)
else()
    set_target_properties(elsa_linalg PROPERTIES LINKER_LANGUAGE CXX)
endif()

write_module_config(${ELSA_MODULE_NAME})
