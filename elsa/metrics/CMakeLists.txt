# list all the headers of the module
set(MODULE_HEADERS RelativeError.h PeakSignaltoNoiseRatio.h)

# list all the code files of the module
set(MODULE_SOURCES RelativeError.cpp PeakSignaltoNoiseRatio.cpp)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging)
list(APPEND MODULE_PRIVATE_DEPS)

# use OpenMP is available
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    list(APPEND MODULE_PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif ()

ADD_ELSA_MODULE(
        metrics "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
        PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

target_include_directories(
        ${ELSA_MODULE_TARGET_NAME} PUBLIC $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

if (ELSA_BUILD_PYTHON_BINDINGS)
    GENERATE_BINDINGS(
            ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
            ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${MODULE_SOURCES}
    )
endif ()

write_module_config(${ELSA_MODULE_NAME} elsa_core elsa_logging)
