# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(
    run-tests-linalg
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-linalg
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'linalg'."
)
add_custom_target(build-tests-linalg)

# the actual tests
ELSA_DOCTEST(Matrix)
ELSA_DOCTEST(Vector)

if(ELSA_CUDA_ENABLED)
    set_source_files_properties(test_Matrix.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(test_Vector.cpp PROPERTIES LANGUAGE CUDA)
endif()
