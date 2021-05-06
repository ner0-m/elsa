# enable ctest and doctest test discovery
include(CTest)
include(doctest)

# Add custom targets for fine building and running
add_custom_target(build-tests-solvers)
add_custom_target(
    run-tests-solvers
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-solvers
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'solvers'."
)

# the actual tests
ELSA_DOCTEST(GradientDescent DEPENDENCIES elsa_generators elsa_projectors)
ELSA_DOCTEST(CG DEPENDENCIES elsa_generators elsa_projectors)
ELSA_DOCTEST(ISTA DEPENDENCIES)
ELSA_DOCTEST(FISTA DEPENDENCIES)
ELSA_DOCTEST(FGM DEPENDENCIES elsa_generators elsa_projectors)
ELSA_DOCTEST(OGM DEPENDENCIES elsa_generators elsa_projectors)
ELSA_DOCTEST(ADMM DEPENDENCIES)
ELSA_DOCTEST(SQS DEPENDENCIES elsa_generators elsa_projectors)
