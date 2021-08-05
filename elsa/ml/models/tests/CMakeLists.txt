# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(
        run-tests-ml-models
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS build-tests-ml-models
        USES_TERMINAL
        COMMENT "Build and run the tests from module 'ml/models'."
)

add_custom_target(build-tests-ml-models)

# the actual tests
ELSA_DOCTEST(AutoEncoder)
