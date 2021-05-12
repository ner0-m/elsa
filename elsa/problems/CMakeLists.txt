# list all the headers of the module
set(MODULE_HEADERS
    RegularizationTerm.h
    Problem.h
    WLSProblem.h
    TikhonovProblem.h
    QuadricProblem.h
    LASSOProblem.h
    SplittingProblem.h
    SubsetProblem.h
    WLSSubsetProblem.h
    JacobiPreconditioner.h
)

# list all the code files of the module
set(MODULE_SOURCES
    RegularizationTerm.cpp
    Problem.cpp
    WLSProblem.cpp
    TikhonovProblem.cpp
    QuadricProblem.cpp
    LASSOProblem.cpp
    SplittingProblem.cpp
    SubsetProblem.cpp
    WLSSubsetProblem.cpp
    JacobiPreconditioner.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging elsa_operators elsa_functionals)
list(APPEND MODULE_PRIVATE_DEPS)

AddElsaModule(
    problems "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

if(ELSA_BUILD_PYTHON_BINDINGS)
    generate_bindings(
        ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
        ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${MODULE_SOURCES}
    )
endif()
 
WriteModuleConfig(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging elsa_operators elsa_functionals)
