# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(build-tests-config)
add_custom_target(
    run-tests-config
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-config
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'config'."
)

# the actual tests
