# list all the headers of the module
set(MODULE_HEADERS CUDADefines.h)

# list all the code files of the module
set(MODULE_SOURCES)

list(APPEND MODULE_PUBLIC_DEPS)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    config "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

set_target_properties(elsa_config PROPERTIES LINKER_LANGUAGE CXX)
write_module_config(${ELSA_MODULE_NAME})
