# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(
    run-tests-storage
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-storage
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'storage'."
)
add_custom_target(build-tests-storage)

# the actual tests
ELSA_DOCTEST(Reductions)
ELSA_DOCTEST(Transforms)
ELSA_DOCTEST(MemoryResources)
ELSA_DOCTEST(ContiguousVector)
ELSA_DOCTEST(AllocationHint)

if(ELSA_CUDA_ENABLED)
    set_source_files_properties(test_Reductions.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(test_Transforms.cpp PROPERTIES LANGUAGE CUDA)
endif()
