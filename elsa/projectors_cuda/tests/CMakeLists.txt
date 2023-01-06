# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(
    run-tests-projectors_cuda
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-projectors_cuda
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'projectors_cuda'."
)

add_custom_target(build-tests-projectors_cuda)

ELSA_DOCTEST(SiddonsMethodCUDA)
ELSA_DOCTEST(JosephsMethodCUDA)
ELSA_DOCTEST(VoxelProjectorCUDA)
