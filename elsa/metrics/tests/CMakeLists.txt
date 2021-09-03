# enable ctest and doctest test discovery
include(CTest)
include(doctest)

# Add custom targets for fine building and running
add_custom_target(build-tests-metrics)
add_custom_target(
    run-tests-metrics
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-metrics
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'metrics'."
)

# the actual tests
ELSA_DOCTEST(MSE)
ELSA_DOCTEST(PSNR)
