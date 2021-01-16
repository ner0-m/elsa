# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(EllipseGenerator)
ELSA_TEST(PhantomGenerator)
ELSA_TEST(CircleTrajectoryGenerator)
