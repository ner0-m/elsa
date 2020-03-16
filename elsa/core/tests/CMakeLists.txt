cmake_minimum_required(VERSION 3.9)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

# the actual tests
ELSA_TEST(elsaDefines)
ELSA_TEST(DataDescriptor)
ELSA_TEST(IdenticalBlocksDescriptor)
ELSA_TEST(PartitionDescriptor)
ELSA_TEST(RandomBlocksDescriptor)
ELSA_TEST(DataContainer)
ELSA_TEST(DataHandlers)
ELSA_TEST(DataHandlerMap)
ELSA_TEST(LinearOperator)
ELSA_TEST(ExpressionTemplates)

if(ELSA_BENCHMARKS)
    ELSA_TEST(BenchmarkExpressionTemplates)
    ELSA_TEST(BenchmarkDataContainerConstruction)
endif()
