# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(build-tests-quickvec)
add_custom_target(
    run-tests-quickvec
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-quickvec
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'quickvec'."
)

ELSA_DOCTEST(SharedPointer)
ELSA_DOCTEST(Vector)
