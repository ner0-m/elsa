#include <catch2/catch.hpp>
#include <iostream>
#include <random>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "ActivationLayer.h"

using namespace elsa;

TEST_CASE("Relu semantics", "elsa_ml")
{
    using Layer = Relu<float, MlBackend::Dnnl>;

    std::mt19937 mt(123); // The random number generator using a deterministic seed
    std::uniform_real_distribution<float> dist(-1, 1);

    IndexVector_t inputVec(4);
    inputVec << 1, 1, 4, 4;
    DataDescriptor inputDesc(inputVec);

    Layer layer(inputDesc);
    float alpha = 0.5f;
    layer.setAlpha(alpha);

    SECTION("Forward semantics")
    {
        DataContainer<float> input(inputDesc);

        for (auto& coeff : input)
            coeff = dist(mt);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();

        auto f = [&alpha](const auto& coeff) {
            if (coeff <= 0)
                return alpha * coeff;
            else
                return coeff;
        };

        for (int i = 0; i < output.getDataDescriptor().getNumberOfCoefficients(); ++i)
            REQUIRE(output[i] == Approx(f(input[i])).epsilon(0.001));
    }

    SECTION("Backward semantics")
    {
        DataContainer<float> input(inputDesc);
        DataContainer<float> outputGradient(inputDesc);

        for (auto& coeff : outputGradient)
            coeff = dist(mt);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->setOutputGradient(outputGradient);
        backend->compile(PropagationKind::Full);
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->backwardPropagate(s);
        s.wait();
        auto inputGradient = backend->getInputGradient();

        auto f_der = [&alpha](const auto& coeff) {
            if (coeff <= 0)
                return alpha;
            else
                return 1.f;
        };

        for (int i = 0; i < inputGradient.getDataDescriptor().getNumberOfCoefficients(); ++i)
            REQUIRE(inputGradient[i] == Approx(f_der(input[i]) * outputGradient[i]).epsilon(0.001));
    }
}

TEST_CASE("Abs semantics", "elsa_ml")
{
    using Layer = Abs<float, MlBackend::Dnnl>;

    std::mt19937 mt(123); // The random number generator using a deterministic seed
    std::uniform_real_distribution<float> dist(-1, 1);

    IndexVector_t inputVec(4);
    inputVec << 1, 1, 4, 4;
    DataDescriptor inputDesc(inputVec);

    Layer layer(inputDesc);

    SECTION("Forward semantics")
    {
        DataContainer<float> input(inputDesc);

        for (auto& coeff : input)
            coeff = dist(mt);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();

        auto f = [](const auto& coeff) { return std::abs(coeff); };

        for (int i = 0; i < output.getDataDescriptor().getNumberOfCoefficients(); ++i)
            REQUIRE(output[i] == Approx(f(input[i])).epsilon(0.001));
    }

    SECTION("Backward semantics")
    {
        DataContainer<float> input(inputDesc);
        DataContainer<float> outputGradient(inputDesc);

        for (auto& coeff : outputGradient)
            coeff = dist(mt);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->setOutputGradient(outputGradient);
        backend->compile(PropagationKind::Full);
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->backwardPropagate(s);
        s.wait();
        auto inputGradient = backend->getInputGradient();

        auto f_der = [](const auto& coeff) {
            if (coeff == 0)
                return 0.f;
            else if (coeff < 0)
                return -1.f;
            else
                return 1.f;
        };

        for (int i = 0; i < inputGradient.getDataDescriptor().getNumberOfCoefficients(); ++i)
            REQUIRE(inputGradient[i] == Approx(f_der(input[i]) * outputGradient[i]).epsilon(0.001));
    }
}

TEST_CASE("Elu semantics", "elsa_ml")
{
    using Layer = Elu<float, MlBackend::Dnnl>;

    std::mt19937 mt(123); // The random number generator using a deterministic seed
    std::uniform_real_distribution<float> dist(-1, 1);

    IndexVector_t inputVec(4);
    inputVec << 1, 1, 4, 4;
    DataDescriptor inputDesc(inputVec);

    Layer layer(inputDesc);
    float alpha = dist(mt);
    layer.setAlpha(alpha);

    SECTION("Forward semantics")
    {
        DataContainer<float> input(inputDesc);

        for (auto& coeff : input)
            coeff = dist(mt);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();

        auto f = [&alpha](const auto& coeff) {
            if (coeff <= 0)
                return alpha * (std::exp(coeff) - 1);
            else
                return coeff;
        };

        for (int i = 0; i < output.getDataDescriptor().getNumberOfCoefficients(); ++i)
            REQUIRE(output[i] == Approx(f(input[i])).epsilon(0.001));
    }

    SECTION("Backward semantics")
    {
        DataContainer<float> input(inputDesc);
        DataContainer<float> outputGradient(inputDesc);

        for (auto& coeff : outputGradient)
            coeff = dist(mt);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->setOutputGradient(outputGradient);
        backend->compile(PropagationKind::Full);
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->backwardPropagate(s);
        s.wait();
        auto inputGradient = backend->getInputGradient();

        auto f_der = [&alpha](const auto& coeff) {
            if (coeff <= 0)
                return alpha * std::exp(coeff);
            else
                return 1.f;
        };

        for (int i = 0; i < inputGradient.getDataDescriptor().getNumberOfCoefficients(); ++i)
            REQUIRE(inputGradient[i] == Approx(f_der(input[i]) * outputGradient[i]).epsilon(0.001));
    }
}

TEST_CASE("Linear semantics", "elsa_ml")
{
    using Layer = Linear<float, MlBackend::Dnnl>;

    std::mt19937 mt(123); // The random number generator using a deterministic seed
    std::uniform_real_distribution<float> dist(-1, 1);

    IndexVector_t inputVec(4);
    inputVec << 1, 1, 4, 4;
    DataDescriptor inputDesc(inputVec);

    Layer layer(inputDesc);
    float alpha = dist(mt);
    float beta = dist(mt);
    layer.setAlpha(alpha);
    layer.setBeta(beta);

    SECTION("Forward semantics")
    {
        DataContainer<float> input(inputDesc);

        for (auto& coeff : input)
            coeff = dist(mt);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();

        auto f = [&alpha, &beta](const auto& coeff) { return alpha * coeff + beta; };

        for (int i = 0; i < output.getDataDescriptor().getNumberOfCoefficients(); ++i)
            REQUIRE(output[i] == Approx(f(input[i])).epsilon(0.001));
    }

    SECTION("Backward semantics")
    {
        DataContainer<float> input(inputDesc);
        DataContainer<float> outputGradient(inputDesc);

        for (auto& coeff : outputGradient)
            coeff = dist(mt);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->setOutputGradient(outputGradient);
        backend->compile(PropagationKind::Full);
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->backwardPropagate(s);
        s.wait();
        auto inputGradient = backend->getInputGradient();

        auto f_der = [&alpha](const auto& coeff) { return alpha; };

        for (int i = 0; i < inputGradient.getDataDescriptor().getNumberOfCoefficients(); ++i)
            REQUIRE(inputGradient[i] == Approx(f_der(input[i]) * outputGradient[i]).epsilon(0.001));
    }
}