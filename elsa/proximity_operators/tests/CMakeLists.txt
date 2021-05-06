# enable ctest and doctest test discovery
include(CTest)
include(doctest)

# Add custom targets for fine building and running
add_custom_target(build-tests-proximity_operators)
add_custom_target(
    run-tests-proximity_operators
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-proximity_operators
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'proximity_operators'."
)

# the actual tests
ELSA_DOCTEST(SoftThresholding)
ELSA_DOCTEST(HardThresholding)
