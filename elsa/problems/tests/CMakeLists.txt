# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(RegularizationTerm)
ELSA_TEST(Problem)
ELSA_TEST(WLSProblem)
ELSA_TEST(QuadricProblem)
ELSA_TEST(TikhonovProblem)
