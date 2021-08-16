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
ELSA_DOCTEST(PlanarDetectorDescriptor)
ELSA_DOCTEST(IdenticalBlocksDescriptor)
ELSA_DOCTEST(PartitionDescriptor)
ELSA_DOCTEST(RandomBlocksDescriptor)
ELSA_DOCTEST(LinearOperator)
ELSA_DOCTEST(StrongTypes)
ELSA_DOCTEST(Geometry)
ELSA_DOCTEST(ExpressionTemplates)
ELSA_DOCTEST(DataHandlers)
ELSA_DOCTEST(DataHandlerMap)
ELSA_DOCTEST(DataContainer)
