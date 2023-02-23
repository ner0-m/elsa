# list all the headers of the module
set(MODULE_HEADERS
    LinearResidual.h
    Functional.h
    L0PseudoNorm.h
    L1Norm.h
    L2Squared.h
    LeastSquares.h
    WeightedL2NormPow2.h
    LInfNorm.h
    Huber.h
    PseudoHuber.h
    Quadric.h
    EmissionLogLikelihood.h
    TransmissionLogLikelihood.h
    Constraint.h
    WeightedL1Norm.h
    ConstantFunctional.h
    IndicatorFunctionals.h
)

# list all the code files of the module
set(MODULE_SOURCES
    LinearResidual.cpp
    Functional.cpp
    L0PseudoNorm.cpp
    L1Norm.cpp
    L2Squared.cpp
    LeastSquares.cpp
    WeightedL2NormPow2.cpp
    LInfNorm.cpp
    Huber.cpp
    PseudoHuber.cpp
    Quadric.cpp
    EmissionLogLikelihood.cpp
    TransmissionLogLikelihood.cpp
    Constraint.cpp
    WeightedL1Norm.cpp
    ConstantFunctional.cpp
    IndicatorFunctionals.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging elsa_operators)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    functionals "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging elsa_operators)
