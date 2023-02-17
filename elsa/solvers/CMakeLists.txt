# list all the headers of the module
set(MODULE_HEADERS
    Solver.h
    GradientDescent.h
    PGD.h
    APGD.h
    CGNE.h
    CGLS.h
    FGM.h
    OGM.h
    ADMML2.h
    SQS.h
    OrthogonalMatchingPursuit.h
    LandweberIteration.h
    Landweber.h
    SIRT.h
    AB_GMRES.h
    BA_GMRES.h
    PowerIterations.h
)

# list all the code files of the module
set(MODULE_SOURCES
    Solver.cpp
    GradientDescent.cpp
    PGD.cpp
    APGD.cpp
    CGNE.cpp
    CGLS.cpp
    FGM.cpp
    OGM.cpp
    ADMML2.cpp
    SQS.cpp
    OrthogonalMatchingPursuit.cpp
    LandweberIteration.cpp
    Landweber.cpp
    SIRT.cpp
    AB_GMRES.cpp
    BA_GMRES.cpp
    PowerIterations.cpp
)

list(
    APPEND
    MODULE_PUBLIC_DEPS
    elsa_core
    elsa_logging
    elsa_proximal_operators
    elsa_operators
    elsa_functionals
)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    solvers "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

write_module_config(
    ${ELSA_MODULE_NAME}
    DEPENDENCIES
    elsa_core
    elsa_logging
    elsa_proximal_operators
    elsa_operators
    elsa_functionals
)
