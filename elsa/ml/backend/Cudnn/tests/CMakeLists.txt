cmake_minimum_required(VERSION 3.10)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

ELSA_TEST(CudnnActivation)
ELSA_TEST(CudnnMemory)
ELSA_TEST(CudnnMerging)
ELSA_TEST(CudnnNoop)
ELSA_TEST(CudnnDeconvolution)
ELSA_TEST(CudnnReshape)
