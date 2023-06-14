# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(
    run-tests-core
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-core
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'core'."
)

add_custom_target(build-tests-core)

ELSA_DOCTEST(elsaDefines)
ELSA_DOCTEST(FFT)
ELSA_DOCTEST(VolumeDescriptor)
ELSA_DOCTEST(CurvedDetectorDescriptor)
ELSA_DOCTEST(PlanarDetectorDescriptor)
ELSA_DOCTEST(IdenticalBlocksDescriptor)
ELSA_DOCTEST(PartitionDescriptor)
ELSA_DOCTEST(RandomBlocksDescriptor)
ELSA_DOCTEST(LinearOperator)
ELSA_DOCTEST(StrongTypes)
ELSA_DOCTEST(Geometry)
ELSA_DOCTEST(DataContainer)
ELSA_DOCTEST(DataContainerFormatter)
ELSA_DOCTEST(CartesianIndices)
ELSA_DOCTEST(Bessel)
ELSA_DOCTEST(Math)
ELSA_DOCTEST(MaybeUninitialized)

if(CUDAToolkit_FOUND)
    target_compile_definitions(
        "${ELSA_MODULE_TARGET_NAME}" PRIVATE ELSA_CUDA_TOOLKIT_PRESENT ELSA_CUFFT_CACHE_SIZE=${ELSA_CUFFT_CACHE_SIZE}
    )
    set_source_files_properties(test_FFT.cpp PROPERTIES LANGUAGE CUDA)
endif()
