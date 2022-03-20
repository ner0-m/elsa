# enable ctest and doctest test discovery
include(CTest)
include(doctest)

# Add target to build tests for module
add_custom_target(build-tests-io)
add_custom_target(
    run-tests-io
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-io
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'io'."
)

# the actual tests
ELSA_DOCTEST(ioUtils)
ELSA_DOCTEST(IO)
ELSA_DOCTEST(EDFHandler)
ELSA_DOCTEST(MHDHandler)
ELSA_DOCTEST(PGMHandler)
