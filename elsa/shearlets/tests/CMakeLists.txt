# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(DiscreteShearletTransform)
ELSA_TEST(ConeAdaptedDiscreteShearletTransform)
