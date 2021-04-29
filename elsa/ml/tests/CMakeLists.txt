cmake_minimum_required(VERSION 3.9)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(Common)
ELSA_TEST(Graph)
ELSA_TEST(CoreLayers)
ELSA_TEST(ConvLayers)
ELSA_TEST(Loss)
ELSA_TEST(Utils)
ELSA_TEST(Reshape)
ELSA_TEST(Initializer)
