# enable ctest and doctest test discovery
include(CTest)
include(doctest)

add_custom_target(
    run-tests-generators
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS build-tests-generators
    USES_TERMINAL
    COMMENT "Build and run the tests from module 'generators'."
)

add_custom_target(build-tests-generators)

# the actual tests
ELSA_DOCTEST(EllipseGenerator)
ELSA_DOCTEST(Phantoms)
ELSA_DOCTEST(CircleTrajectoryGenerator)
ELSA_DOCTEST(LimitedAngleTrajectoryGenerator)
ELSA_DOCTEST(SphereTrajectoryGenerator)
ELSA_DOCTEST(NoiseGenerators)
ELSA_DOCTEST(Sphere)
ELSA_DOCTEST(Box)
