cmake_minimum_required(VERSION 3.9)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(GradientDescent DEPENDENCIES elsa_generators elsa_projectors)
ELSA_TEST(CG DEPENDENCIES elsa_generators elsa_projectors)
ELSA_TEST(ISTA DEPENDENCIES)
ELSA_TEST(FISTA DEPENDENCIES)
ELSA_TEST(FGM DEPENDENCIES elsa_generators elsa_projectors)
ELSA_TEST(OGM DEPENDENCIES elsa_generators elsa_projectors)
ELSA_TEST(ADMM DEPENDENCIES)
ELSA_TEST(SQS DEPENDENCIES elsa_generators elsa_projectors)
