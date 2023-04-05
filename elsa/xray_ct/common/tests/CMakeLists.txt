# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(
    run-tests-xray-common
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-xray
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'xray-common'."
)

add_custom_target(build-tests-xray-common)

ELSA_DOCTEST(CurvedDetectorDescriptor)
ELSA_DOCTEST(PlanarDetectorDescriptor)
ELSA_DOCTEST(Geometry)
