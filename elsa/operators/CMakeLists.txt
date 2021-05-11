# list all the headers of the module
set(MODULE_HEADERS Identity.h Scaling.h FiniteDifferences.h BlockLinearOperator.h)

# list all the code files of the module
set(MODULE_SOURCES Identity.cpp Scaling.cpp FiniteDifferences.cpp BlockLinearOperator.cpp)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging)
list(APPEND MODULE_PRIVATE_DEPS)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    list(APPEND MODULE_PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif()

AddElsaModule(
    operators "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

if(ELSA_BUILD_PYTHON_BINDINGS)
    generate_bindings(
        ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
        ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${MODULE_SOURCES}
    )
endif()
