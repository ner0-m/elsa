cmake_minimum_required(VERSION 3.9)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

ELSA_TEST(DnnlActivation)
ELSA_TEST(DnnlPooling)
ELSA_TEST(DnnlMerging)