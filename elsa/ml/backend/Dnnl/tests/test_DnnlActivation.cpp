/**
 * \file test_common.cpp
 *
 * \brief Tests for common ml functionality
 *
 * \author David Tellenbach
 */

#include <catch2/catch.hpp>
#include <random>

#include "DataContainer.h"
#include "DnnlActivationLayer.h"

using namespace elsa;
using namespace elsa::ml::detail;

template <typename F, typename data_t>
static void testActivation(F func, data_t alpha, data_t beta, const DataContainer<data_t>& input,
                           const DataContainer<data_t>& test)
{
    REQUIRE(input.getSize() == test.getSize());

    for (int i = 0; i < input.getSize(); ++i) {
        REQUIRE(func(input[i], alpha, beta) == Approx(test[i]));
    }
}

template <typename F, typename data_t>
static void testActivationDerivative(F func, data_t alpha, data_t beta,
                                     const DataContainer<data_t>& input,
                                     const DataContainer<data_t>& outGrad,
                                     const DataContainer<data_t>& test)
{
    REQUIRE(input.getSize() == test.getSize());

    for (int i = 0; i < input.getSize(); ++i) {
        REQUIRE(func(input[i], alpha, beta) * outGrad[i] == Approx(test[i]));
    }
}

template <typename LayerType, typename Func, typename FuncDer>
void testActivationLayer(Func f, FuncDer f_der)
{
    std::mt19937 mt(123); // The random number generator using a deterministic seed
    std::uniform_real_distribution<float> dist(-1, 1);
    std::uniform_int_distribution<index_t> distIdx(1, 10);

    // Create random input
    IndexVector_t inputVec(4);
    for (int i = 0; i < inputVec.size(); ++i)
        inputVec[i] = distIdx(mt);

    VolumeDescriptor inputDesc(inputVec);

    DataContainer<float> input(inputDesc);
    for (auto& coeff : input)
        coeff = dist(mt);

    // Create random output-gradient
    DataContainer<float> outputGradient(inputDesc);
    for (auto& coeff : outputGradient)
        coeff = dist(mt);

    // Construct layer and set parameters
    LayerType layer(inputDesc);
    float alpha = dist(mt);
    layer.setAlpha(alpha);

    float beta = dist(mt);
    layer.setBeta(beta);

    // Set input and compile layer
    layer.setInput(input);
    layer.compile(elsa::ml::PropagationKind::Full);

    // Get Dnnl execution stream
    auto engine = layer.getEngine();
    dnnl::stream s(*engine);

    SECTION("Basics")
    {
        REQUIRE(layer.getInputDescriptor() == inputDesc);
        REQUIRE(layer.getOutputDescriptor() == inputDesc);
    }
    SECTION("Forward")
    {
        layer.forwardPropagate(s);
        auto output = layer.getOutput();

        testActivation(f, alpha, beta, input, output);
    }

    SECTION("Backward")
    {
        layer.setOutputGradient(outputGradient);
        layer.backwardPropagate(s);
        auto inputGradient = layer.getInputGradient();

        testActivationDerivative(f_der, alpha, beta, input, outputGradient, inputGradient);
    }
}

TEST_CASE("DnnlRelu", "[ml][dnnl]")
{
    auto f = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                [[maybe_unused]] float beta) {
        if (coeff <= 0)
            return alpha * coeff;
        else
            return coeff;
    };

    auto f_der = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                    [[maybe_unused]] float beta) {
        if (coeff <= 0)
            return alpha;
        else
            return 1.f;
    };
    testActivationLayer<DnnlRelu<float>>(f, f_der);
}

TEST_CASE("DnnlAbs", "[ml][dnnl]")
{
    auto f = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                [[maybe_unused]] float beta) { return std::abs(coeff); };

    auto f_der = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                    [[maybe_unused]] float beta) {
        if (coeff == 0)
            return 0.f;
        else if (coeff < 0)
            return -1.f;
        else
            return 1.f;
    };
    testActivationLayer<DnnlAbs<float>>(f, f_der);
}

TEST_CASE("DnnlElu", "[ml][dnnl]")
{
    auto f = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                [[maybe_unused]] float beta) {
        if (coeff <= 0)
            return alpha * (std::exp(coeff) - 1);
        else
            return coeff;
    };

    auto f_der = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                    [[maybe_unused]] float beta) {
        if (coeff <= 0)
            return alpha * std::exp(coeff);
        else
            return 1.f;
    };
    testActivationLayer<DnnlElu<float>>(f, f_der);
}

TEST_CASE("DnnlLinear", "[ml][dnnl]")
{
    auto f = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                [[maybe_unused]] float beta) { return alpha * coeff + beta; };

    auto f_der = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                    [[maybe_unused]] float beta) { return alpha; };
    testActivationLayer<DnnlLinear<float>>(f, f_der);
}

TEST_CASE("DnnlTanh", "[ml][dnnl]")
{
    auto f = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                [[maybe_unused]] float beta) { return std::tanh(coeff); };

    auto f_der = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                    [[maybe_unused]] float beta) {
        return (1 - std::tanh(coeff) * std::tanh(coeff));
    };
    testActivationLayer<DnnlTanh<float>>(f, f_der);
}

TEST_CASE("DnnlLogistic", "[ml][dnnl]")
{
    auto f = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                [[maybe_unused]] float beta) { return 1.f / (1.f + std::exp(-1.f * coeff)); };

    auto f_der = [&f]([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                      [[maybe_unused]] float beta) {
        return f(coeff, alpha, beta) * (1.f - f(coeff, alpha, beta));
    };
    testActivationLayer<DnnlLogistic<float>>(f, f_der);
}

TEST_CASE("DnnlExp", "[ml][dnnl]")
{
    auto f = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                [[maybe_unused]] float beta) { return std::exp(coeff); };

    auto f_der = [&f]([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                      [[maybe_unused]] float beta) { return f(coeff, alpha, beta); };
    testActivationLayer<DnnlExp<float>>(f, f_der);
}

TEST_CASE("DnnlSoftRelu", "[ml][dnnl]")
{
    auto f = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                [[maybe_unused]] float beta) { return std::log(1 + std::exp(coeff)); };

    auto f_der = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha,
                    [[maybe_unused]] float beta) { return 1.f / (1.f + std::exp(-1.f * coeff)); };
    testActivationLayer<DnnlSoftRelu<float>>(f, f_der);
}
