# enable ctest and doctest test discovery
include(CTest)
include(doctest)

# add custom targets for fine building and running
add_custom_target(build-tests-tasks)
add_custom_target(
        run-tests-tasks
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS build-tests-tasks
        USES_TERMINAL
        COMMENT "Build and run the tests from module 'tasks'."
)

# the actual tests
# ELSA_DOCTEST(InpaintLimitedAngleSingularitiesTask PUBLIC_DEPENDENCIES elsa_problems elsa_solvers)
ELSA_DOCTEST(InpaintLimitedAngleSingularitiesTask)
