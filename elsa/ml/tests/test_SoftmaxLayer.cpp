#include <catch2/catch.hpp>
#include <type_traits>
#include <random>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "SoftmaxLayer.h"

using namespace elsa;

TEST_CASE("SoftmaxLayer", "elsa_ml")
{
    using Layer = SoftmaxLayer<float, MlBackend::Dnnl>;

    SECTION("Test 1")
    {
        Eigen::VectorXf inputValues = Eigen::VectorXf::Random(10);
        IndexVector_t inputVec(2);
        inputVec << 1, 10;
        DataDescriptor inputDesc(inputVec);

        Layer layer(inputDesc);

        DataContainer<float> input(inputDesc, inputValues);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();

        REQUIRE(output.sum() == Approx(1));
    }

    SECTION("Test 2")
    {
        Eigen::VectorXf inputValues = Eigen::VectorXf::Random(40);
        IndexVector_t inputVec(3);
        inputVec << 2, 10, 2;
        DataDescriptor inputDesc(inputVec);

        Layer layer(inputDesc);

        DataContainer<float> input(inputDesc, inputValues);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();

        REQUIRE(output.sum() == Approx(4));
    }

    SECTION("Test 3")
    {
        Eigen::VectorXf inputValues(7);
        inputValues << 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f;
        IndexVector_t inputVec(2);
        inputVec << 1, 7;
        DataDescriptor inputDesc(inputVec);

        Layer layer(inputDesc);

        DataContainer<float> input(inputDesc, inputValues);

        auto backend = layer.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();

        REQUIRE(output.sum() == Approx(1));

        REQUIRE(output[0] == Approx(0.02364054f));
        REQUIRE(output[1] == Approx(0.06426166f));
        REQUIRE(output[2] == Approx(0.17468130f));
        REQUIRE(output[3] == Approx(0.47483300f));
        REQUIRE(output[4] == Approx(0.02364054f));
        REQUIRE(output[5] == Approx(0.06426166f));
        REQUIRE(output[6] == Approx(0.17468130f));
    }
}
