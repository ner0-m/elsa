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
ELSA_DOCTEST(GradientDescent)
ELSA_DOCTEST(PGD)
ELSA_DOCTEST(APGD)
ELSA_DOCTEST(CGLS)
ELSA_DOCTEST(CGNE)
ELSA_DOCTEST(FGM)
ELSA_DOCTEST(OGM)
ELSA_DOCTEST(ADMML2)
ELSA_DOCTEST(SQS PUBLIC_DEPENDENCIES elsa_projectors)
ELSA_DOCTEST(OrthogonalMatchingPursuit)
ELSA_DOCTEST(Landweber)
ELSA_DOCTEST(SIRT)
ELSA_DOCTEST(AB_GMRES PUBLIC_DEPENDENCIES elsa_generators elsa_projectors)
ELSA_DOCTEST(BA_GMRES PUBLIC_DEPENDENCIES elsa_generators elsa_projectors)
ELSA_DOCTEST(PowerIterations)
ELSA_DOCTEST(RegularizedInversion)
