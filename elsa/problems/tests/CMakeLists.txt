cmake_minimum_required(VERSION 3.9)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(WLSProblem)
ELSA_TEST(RegularizationTerm)
ELSA_TEST(OptimizationProblem)