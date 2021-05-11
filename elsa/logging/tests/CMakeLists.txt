# enable ctest and doctest test discovery
include(CTest)
include(doctest)

# Add custom targets for fine building and running
add_custom_target(build-tests-logging)
add_custom_target(
    run-tests-logging
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-logging
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'logging'."
)

# the actual tests
ELSA_DOCTEST(Logger)
ELSA_DOCTEST(LogGuard)
ELSA_DOCTEST(Timer)
