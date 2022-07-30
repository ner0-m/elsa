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
    RepresentationProblem.h
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
    RepresentationProblem.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging elsa_operators elsa_functionals)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    problems "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging elsa_operators elsa_functionals)
