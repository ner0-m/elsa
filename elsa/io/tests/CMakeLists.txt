cmake_minimum_required(VERSION 3.9)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(ioUtils)
ELSA_TEST(EDFHandler)
ELSA_TEST(MHDHandler)
