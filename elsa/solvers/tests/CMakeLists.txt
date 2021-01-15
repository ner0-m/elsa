cmake_minimum_required(VERSION 3.9)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(GradientDescent)
ELSA_TEST(CG)
ELSA_TEST(FGM)
ELSA_TEST(OGM)