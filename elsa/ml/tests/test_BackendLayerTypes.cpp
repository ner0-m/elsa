#include <catch2/catch.hpp>
#include <type_traits>

#include "elsaDefines.h"
#include "ActivationLayer.h"
#include "ConvLayer.h"
#include "DenseLayer.h"
#include "PoolingLayer.h"
#include "SoftmaxLayer.h"

using namespace elsa;

TEMPLATE_TEST_CASE("BackendLayerTypes", "elsa_ml", float)
{
    SECTION("MlBackend::Dnnl")
    {
        constexpr MlBackend Backend = MlBackend::Dnnl;

        // clang-format off
        REQUIRE(std::is_same_v<typename Relu         <TestType, Backend>::BackendLayerType, DnnlRelu<TestType>>);
        REQUIRE(std::is_same_v<typename ConvLayer    <TestType, Backend>::BackendLayerType, DnnlConvLayer<TestType>>);
        REQUIRE(std::is_same_v<typename DenseLayer   <TestType, Backend>::BackendLayerType, DnnlDenseLayer<TestType>>);
        REQUIRE(std::is_same_v<typename PoolingLayer <TestType, Backend>::BackendLayerType, DnnlPoolingLayer<TestType>>);
        REQUIRE(std::is_same_v<typename SoftmaxLayer <TestType, Backend>::BackendLayerType, DnnlSoftmaxLayer<TestType>>);
        // clang-format on
    }
}
