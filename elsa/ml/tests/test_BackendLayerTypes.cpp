#include <catch2/catch.hpp>
#include <type_traits>

#include "elsaDefines.h"
#include "Activation.h"
#include "Conv.h"
#include "Dense.h"
#include "Pooling.h"

using namespace elsa;

TEMPLATE_TEST_CASE("BackendLayerTypes", "elsa_ml", float)
{
    SECTION("MlBackend::Dnnl")
    {
        constexpr MlBackend Backend = MlBackend::Dnnl;

        REQUIRE(
            std::is_same_v<typename Relu<TestType, Backend>::BackendLayerType, DnnlRelu<TestType>>);
        REQUIRE(
            std::is_same_v<typename Conv<TestType, Backend>::BackendLayerType, DnnlConv<TestType>>);
        REQUIRE(std::is_same_v<typename Dense<TestType, Backend>::BackendLayerType,
                               DnnlDense<TestType>>);
        REQUIRE(std::is_same_v<typename Pooling<TestType, Backend>::BackendLayerType,
                               DnnlPooling<TestType>>);
    }
}
