cmake_minimum_required(VERSION 3.9)

# list all the headers of the module
set(MODULE_HEADERS
        Residual.h
        LinearResidual.h
        Functional.h
        L1Norm.h
        L2NormPow2.h
        WeightedL2NormPow2.h
        LInfNorm.h
        Huber.h
        PseudoHuber.h
        Quadric.h
        EmissionLogLikelihood.h
        TransmissionLogLikelihood.h)

# list all the code files of the module
set(MODULE_SOURCES
        Residual.cpp
        LinearResidual.cpp
        Functional.cpp
        L1Norm.cpp
        L2NormPow2.cpp
        WeightedL2NormPow2.cpp
        LInfNorm.cpp
        Huber.cpp
        PseudoHuber.cpp
        Quadric.cpp
        EmissionLogLikelihood.cpp
        TransmissionLogLikelihood.cpp)

# setup library see elsa/CMakeLists.txt
ELSA_LIBRARY_SETUP(functionals elsa_core elsa_logging elsa_operators)
