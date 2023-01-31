# enable ctest and doctest test discovery
include(CTest)
include(doctest)

# Add custom targets for fine building and running
add_custom_target(build-tests-proximal_operators)
add_custom_target(
    run-tests-proximal_operators
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-proximal_operators
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'proximal_operators'."
)

# the actual tests
ELSA_DOCTEST(ProximalL1)
ELSA_DOCTEST(ProximalL0)
ELSA_DOCTEST(ProximalBoxConstraint)
