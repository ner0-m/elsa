cmake_minimum_required(VERSION 3.9)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(DataDescriptor)
ELSA_TEST(BlockDescriptor)
ELSA_TEST(DataContainer)
ELSA_TEST(DataHandlerCPU)
ELSA_TEST(LinearOperator)
