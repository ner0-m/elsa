# enable ctest and Catch test discovery
include(CTest)
include(doctest)

# Add custom targets for fine building and running
add_custom_target(build-tests-shearlets)
add_custom_target(
        run-tests-shearlets
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS build-tests-shearlets
        USES_TERMINAL
        COMMENT "Build and run the tests from module 'shearlets'."
)

# the actual tests
ELSA_DOCTEST(ShearletTransform)
