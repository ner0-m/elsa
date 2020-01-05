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

    SECTION("Backend semantics")
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
}

TEST_CASE("Abs semantics", "elsa_ml")
{
    using Layer = Abs<float, MlBackend::Dnnl>;

    IndexVector_t inputVec(4);
    inputVec << 1, 1, 4, 4;
    DataDescriptor inputDesc(inputVec);

    Layer layer(inputDesc);

    SECTION("Backend semantics")
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

    SECTION("Backend semantics")
    {
        Layer layer(inputDesc);
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
}