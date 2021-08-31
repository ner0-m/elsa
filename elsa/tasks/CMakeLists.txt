# list all the headers of the module
set(MODULE_HEADERS
    ImageDenoisingTask.h
)

# list all the code files of the module
set(MODULE_SOURCES
    ImageDenoisingTask.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging elsa_problems elsa_proximity_operators elsa_solvers)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    tasks "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

if(ELSA_BUILD_PYTHON_BINDINGS)
    GENERATE_BINDINGS(
        ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
        ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${MODULE_SOURCES}
    )
endif()
 
write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging elsa_problems elsa_proximity_operators)
