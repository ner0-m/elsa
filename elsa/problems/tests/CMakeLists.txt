# enable ctest and doctest test discovery
include(CTest)
include(doctest)

# Add custom targets for fine building and running
add_custom_target(build-tests-problems)
add_custom_target(
    run-tests-problems
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-problems
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'problems'."
)

# the actual tests
ELSA_DOCTEST(RegularizationTerm)
ELSA_DOCTEST(Problem)
ELSA_DOCTEST(WLSProblem)
ELSA_DOCTEST(QuadricProblem)
ELSA_DOCTEST(TikhonovProblem)
ELSA_DOCTEST(LASSOProblem)
ELSA_DOCTEST(SplittingProblem)
ELSA_DOCTEST(SubsetProblem)
ELSA_DOCTEST(WLSSubsetProblem)
ELSA_DOCTEST(JacobiPreconditioner)
ELSA_DOCTEST(RepresentationProblem)
ELSA_DOCTEST(PowerIterations)
