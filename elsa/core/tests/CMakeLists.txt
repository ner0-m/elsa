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
ELSA_DOCTEST(VolumeDescriptor)
ELSA_DOCTEST(IdenticalBlocksDescriptor)
ELSA_DOCTEST(PartitionDescriptor)
ELSA_DOCTEST(RandomBlocksDescriptor)
ELSA_DOCTEST(LinearOperator)
ELSA_DOCTEST(StrongTypes)
ELSA_DOCTEST(DataContainer)
ELSA_DOCTEST(DataContainerFormatter)
ELSA_DOCTEST(CartesianIndices)
ELSA_DOCTEST(Bessel)
ELSA_DOCTEST(Math)
ELSA_DOCTEST(MaybeUninitialized)
