# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(
    run-tests-projectors
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-projectors
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'projectors'."
)

add_custom_target(build-tests-projectors)

# the actual tests
ELSA_DOCTEST(BoundingBox)
ELSA_DOCTEST(Intersection)
ELSA_DOCTEST(TraverseAABB)
ELSA_DOCTEST(TraverseAABBBranchless)
ELSA_DOCTEST(SliceTraversal)
ELSA_DOCTEST(LutProjector)
ELSA_DOCTEST(Blobs)
ELSA_DOCTEST(BlobDerivative)
ELSA_DOCTEST(BSplines)
ELSA_DOCTEST(Luts)
ELSA_DOCTEST(BinaryMethod)
ELSA_DOCTEST(SiddonsMethod)
ELSA_DOCTEST(SiddonsMethodBranchless)
ELSA_DOCTEST(JosephsMethod)
ELSA_DOCTEST(SubsetSampler PUBLIC_DEPENDENCIES elsa_generators)
