# list all the headers of the module
set(MODULE_HEADERS
    Residual.h
    LinearResidual.h
    Functional.h
    L0PseudoNorm.h
    L1Norm.h
    L2NormPow2.h
    WeightedL2NormPow2.h
    LInfNorm.h
    Huber.h
    PseudoHuber.h
    Quadric.h
    EmissionLogLikelihood.h
    TransmissionLogLikelihood.h
    Constraint.h
    WeightedL1Norm.h
    Indicator.h
)

# list all the code files of the module
set(MODULE_SOURCES
    Residual.cpp
    LinearResidual.cpp
    Functional.cpp
    L0PseudoNorm.cpp
    L1Norm.cpp
    L2NormPow2.cpp
    WeightedL2NormPow2.cpp
    LInfNorm.cpp
    Huber.cpp
    PseudoHuber.cpp
    Quadric.cpp
    EmissionLogLikelihood.cpp
    TransmissionLogLikelihood.cpp
    Constraint.cpp
    WeightedL1Norm.cpp
    Indicator.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging elsa_operators)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    functionals "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

if(ELSA_BUILD_PYTHON_BINDINGS)
    GENERATE_BINDINGS(
        ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
        ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${MODULE_SOURCES}
    )
endif()
 
write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging elsa_operators)
