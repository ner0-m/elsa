#include <catch2/catch.hpp>
#include <type_traits>

#include "elsaDefines.h"
#include "Activation.h"
#include "Conv.h"

using namespace elsa;

TEMPLATE_TEST_CASE("BackendLayerTypes", "elsa_ml", float)
{
    SECTION("Relu")
    {
        REQUIRE(std::is_same_v<typename Relu<TestType, MlBackend::Dnnl>::BackendLayerType,
                               DnnlRelu<TestType>>);
    }
    SECTION("Conv")
    {
        REQUIRE(std::is_same_v<typename Conv<TestType, MlBackend::Dnnl>::BackendLayerType,
                               DnnlConv<TestType>>);
    }
}
