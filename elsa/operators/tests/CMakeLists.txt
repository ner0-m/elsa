# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(Identity)
ELSA_TEST(Scaling)
ELSA_TEST(FiniteDifferences)
ELSA_TEST(BlockLinearOperator)
ELSA_TEST(ProjectionOperator)
ELSA_TEST(ParabolicScalingOperator)
ELSA_TEST(ShearingOperator)
