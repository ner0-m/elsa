# list all the headers of the module
set(MODULE_HEADERS
    Solver.h
    GradientDescent.h
    CG.h
    ISTA.h
    FISTA.h
    FGM.h
    OGM.h
    ADMM.h
    SQS.h
    OrthogonalMatchingPursuit.h
    SIRT.h
)

# list all the code files of the module
set(MODULE_SOURCES
    Solver.cpp
    GradientDescent.cpp
    CG.cpp
    ISTA.cpp
    FISTA.cpp
    FGM.cpp
    OGM.cpp
    SQS.cpp
    OrthogonalMatchingPursuit.cpp
    SIRT.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging elsa_problems elsa_proximity_operators)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    solvers "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

if(ELSA_BUILD_PYTHON_BINDINGS)
    GENERATE_BINDINGS(
        ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
        ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${MODULE_SOURCES}
    )
endif()
 
write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging elsa_problems elsa_proximity_operators)
