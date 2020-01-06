#include <catch2/catch.hpp>
#include <type_traits>
#include <random>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "ActivationLayer.h"

using namespace elsa;

TEST_CASE("Relu semantics", "elsa_ml")
{
    using Layer = Relu<float, MlBackend::Dnnl>;

    IndexVector_t inputVec(4);
    inputVec << 1, 1, 4, 4;
    DataDescriptor inputDesc(inputVec);

    Layer layer(inputDesc);

    SECTION("Forward semantics")
    {
        std::mt19937 mt(123); // The random number generator using a deterministic seed
        std::uniform_real_distribution<float> dist;

        DataContainer<float> input(inputDesc);

        for (int i = 0; i < 16; ++i)
            input[i] = dist(mt);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();

        for (int i = 0; i < 16; ++i)
            REQUIRE(output[i] == (input[i] < 0 ? Approx(0.0f) : Approx(input[i])));
    }

    SECTION("Backward semantics")
    {
        std::mt19937 mt(123); // The random number generator using a deterministic seed
        std::uniform_real_distribution<float> dist;

        DataContainer<float> input(inputDesc);
        DataContainer<float> outputGradient(inputDesc);

        for (int i = 0; i < 16; ++i) {
            input[i] = dist(mt);
            outputGradient = dist(mt);
        }

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->setOutputGradient(input);
        std::static_pointer_cast<typename decltype(layer)::BackendLayerType>(backend)->setAlpha(
            .5f);
        backend->compile(PropagationKind::Full);
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->backwardPropagate(s);
        s.wait();
        auto inputGradient = backend->getInputGradient();

        for (int i = 0; i < 16; ++i)
            REQUIRE(inputGradient[i]
                    == (outputGradient[i] < 0 ? Approx(.5f * input[i]) : Approx(input[i])));
    }
}

TEST_CASE("Abs semantics", "elsa_ml")
{
    using Layer = Abs<float, MlBackend::Dnnl>;

    IndexVector_t inputVec(4);
    inputVec << 1, 1, 4, 4;
    DataDescriptor inputDesc(inputVec);

    Layer layer(inputDesc);

    SECTION("Forward semantics")
    {
        std::mt19937 mt(123); // The random number generator using a deterministic seed
        std::uniform_real_distribution<float> dist;

        DataContainer<float> input(inputDesc);

        for (int i = 0; i < 16; ++i)
            input[i] = dist(mt);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();

        for (int i = 0; i < 16; ++i)
            REQUIRE(output[i] == std::abs(input[i]));
    }
    SECTION("Backward semantics")
    {
        std::mt19937 mt(123); // The random number generator using a deterministic seed
        std::uniform_real_distribution<float> dist;

        DataContainer<float> input(inputDesc);
        DataContainer<float> outputGradient(inputDesc);

        for (int i = 0; i < 16; ++i) {
            input[i] = dist(mt);
            outputGradient = dist(mt);
        }

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->setOutputGradient(input);
        backend->compile(PropagationKind::Full);
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->backwardPropagate(s);
        s.wait();
        auto inputGradient = backend->getInputGradient();

        for (int i = 0; i < 16; ++i)
            REQUIRE(inputGradient[i]
                    == (outputGradient[i] < 0 ? Approx(-1.f * input[i]) : Approx(input[i])));
    }
}

TEST_CASE("Elu semantics", "elsa_ml")
{
    using Layer = Elu<float, MlBackend::Dnnl>;

    IndexVector_t inputVec(4);
    inputVec << 3, 2, 4, 4;
    DataDescriptor inputDesc(inputVec);

    std::mt19937 mt(123); // The random number generator using a deterministic seed
    std::uniform_real_distribution<float> dist;

    DataContainer<float> input(inputDesc);

    for (auto& coeff : input)
        coeff = dist(mt);

    Layer layer(inputDesc);

    SECTION("Forward semantics")
    {
        layer.setAlpha(2.f);
        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();

        for (int i = 0; i < inputDesc.getNumberOfCoefficients(); ++i)
            REQUIRE(output[i]
                    == (input[i] < 0 ? Approx(2.f * (std::exp(input[i]) - 1)) : Approx(input[i])));
    }

    SECTION("Backward semantics")
    {
        std::mt19937 mt(123); // The random number generator using a deterministic seed
        std::uniform_real_distribution<float> dist;

        DataContainer<float> input(inputDesc);
        DataContainer<float> outputGradient(inputDesc);

        for (int i = 0; i < 16; ++i) {
            input[i] = dist(mt);
            outputGradient = dist(mt);
        }

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->setOutputGradient(input);
        backend->compile(PropagationKind::Full);
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->backwardPropagate(s);
        s.wait();
        auto inputGradient = backend->getInputGradient();

        for (int i = 0; i < 16; ++i)
            REQUIRE(inputGradient[i]
                    == (outputGradient[i] < 0 ? Approx(2.f * std::exp(input[i]) * input[i])
                                              : Approx(input[i])));
    }
}