#include <catch2/catch.hpp>
#include <type_traits>

#include "elsaDefines.h"
#include "ActivationLayer.h"
#include "ConvLayer.h"
#include "PoolingLayer.h"
#include "DenseLayer.h"
#include "SoftmaxLayer.h"

using namespace elsa;

TEST_CASE("BackendSelector", "elsa_ml")
{
    // clang-format off
    REQUIRE(std::is_same_v<typename detail::BackendSelector<Abs          <float, MlBackend::Dnnl>>::Type, DnnlAbs<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<BoundedRelu  <float, MlBackend::Dnnl>>::Type, DnnlBoundedRelu<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<Elu          <float, MlBackend::Dnnl>>::Type, DnnlElu<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<Exp          <float, MlBackend::Dnnl>>::Type, DnnlExp<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<Gelu         <float, MlBackend::Dnnl>>::Type, DnnlGelu<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<Linear       <float, MlBackend::Dnnl>>::Type, DnnlLinear<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<Logistic     <float, MlBackend::Dnnl>>::Type, DnnlLogistic<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<Relu         <float, MlBackend::Dnnl>>::Type, DnnlRelu<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<SoftRelu     <float, MlBackend::Dnnl>>::Type, DnnlSoftRelu<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<Sqrt         <float, MlBackend::Dnnl>>::Type, DnnlSqrt<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<Square       <float, MlBackend::Dnnl>>::Type, DnnlSquare<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<Swish        <float, MlBackend::Dnnl>>::Type, DnnlSwish<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<Tanh         <float, MlBackend::Dnnl>>::Type, DnnlTanh<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<ConvLayer    <float, MlBackend::Dnnl>>::Type, DnnlConvLayer<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<PoolingLayer <float, MlBackend::Dnnl>>::Type, DnnlPoolingLayer<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<DenseLayer   <float, MlBackend::Dnnl>>::Type, DnnlDenseLayer<float>>);
    REQUIRE(std::is_same_v<typename detail::BackendSelector<SoftmaxLayer <float, MlBackend::Dnnl>>::Type, DnnlSoftmaxLayer<float>>);
    // clang-format on
}
