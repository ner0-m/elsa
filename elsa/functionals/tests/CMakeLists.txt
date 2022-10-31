# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(
    run-tests-functionals
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-functionals
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'functionals'."
)

add_custom_target(build-tests-functionals)

# the actual tests
ELSA_DOCTEST(LinearResidual)
ELSA_DOCTEST(L0PseudoNorm)
ELSA_DOCTEST(L1Norm)
ELSA_DOCTEST(L2NormPow2)
ELSA_DOCTEST(WeightedL2NormPow2)
ELSA_DOCTEST(LInfNorm)
ELSA_DOCTEST(Huber)
ELSA_DOCTEST(PseudoHuber)
ELSA_DOCTEST(Quadric)
ELSA_DOCTEST(EmissionLogLikelihood)
ELSA_DOCTEST(TransmissionLogLikelihood)
ELSA_DOCTEST(Constraint)
ELSA_DOCTEST(WeightedL1Norm)
ELSA_DOCTEST(GibbsUtils)
