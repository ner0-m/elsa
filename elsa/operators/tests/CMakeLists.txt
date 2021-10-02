# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(build-tests-operators)
add_custom_target(
    run-tests-operators
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-operators
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'operators'."
)

# the actual tests
ELSA_DOCTEST(Identity)
ELSA_DOCTEST(Scaling)
ELSA_DOCTEST(FiniteDifferences)
ELSA_DOCTEST(FourierTransform)
ELSA_DOCTEST(BlockLinearOperator)
ELSA_DOCTEST(Dictionary)
ELSA_DOCTEST(ShearletTransform)
