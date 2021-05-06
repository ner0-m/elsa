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
elsa_doctest(Identity)
elsa_doctest(Scaling)
elsa_doctest(FiniteDifferences)
elsa_doctest(BlockLinearOperator)
