cmake_minimum_required(VERSION 3.9)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

ELSA_TEST(BackendLayerTypes)
ELSA_TEST(LayerOutputShape)
ELSA_TEST(BackendSelector)
ELSA_TEST(Activation)
ELSA_TEST(Pooling)
ELSA_TEST(ConvLayer)
ELSA_TEST(DenseLayer)
ELSA_TEST(SequentialNetworkInference)
ELSA_TEST(SoftmaxLayer)
