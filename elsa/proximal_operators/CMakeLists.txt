# list all the headers of the module
set(MODULE_HEADERS
    ProximalOperator.h
    ProximalL1.h
    ProximalL0.h
    ProximalBoxConstraint.h
    CombinedProximal.h
    ProximalL2Squared.h
    ProximalHuber.cpp
)

# list all the code files of the module
set(MODULE_SOURCES
    ProximalOperator.cpp
    ProximalL1.cpp
    ProximalL0.cpp
    ProximalBoxConstraint.cpp
    CombinedProximal.cpp
    ProximalL2Squared.cpp
    ProximalHuber.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging)
list(APPEND MODULE_PRIVATE_DEPS)

# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    list(APPEND MODULE_PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif()

ADD_ELSA_MODULE(
    proximal_operators "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

target_include_directories(
    ${ELSA_MODULE_TARGET_NAME} PUBLIC $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}>
                                      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

write_module_config(${ELSA_MODULE_NAME} elsa_core elsa_logging)
