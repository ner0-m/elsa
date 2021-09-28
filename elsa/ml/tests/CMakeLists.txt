# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(
    run-tests-ml
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-ml
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'ml'."
)

add_custom_target(build-tests-ml)

# the actual tests
ELSA_DOCTEST(Common)
ELSA_DOCTEST(Graph)
ELSA_DOCTEST(CoreLayers)
ELSA_DOCTEST(ConvLayers)
ELSA_DOCTEST(Loss)
ELSA_DOCTEST(Utils)
ELSA_DOCTEST(Reshape)
ELSA_DOCTEST(Initializer)
ELSA_DOCTEST(PhantomNet)
