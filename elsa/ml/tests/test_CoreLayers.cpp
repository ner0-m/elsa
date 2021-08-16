#include "doctest/doctest.h"
#include "Input.h"
#include "Dense.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("ml");

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_CASE_TEMPLATE("CoreLayers", TestType, float)
{
    SECTION("Input")
    {
        IndexVector_t dims{{1, 2, 3}};
        VolumeDescriptor desc(dims);
        auto i = ml::Input<TestType>(desc, 10, "input");
        REQUIRE(i.getInputDescriptor() == desc);
        i.computeOutputDescriptor();
        REQUIRE(i.getOutputDescriptor() == desc);
        REQUIRE(i.getLayerType() == ml::LayerType::Input);
        REQUIRE(i.getBatchSize() == 10);
        REQUIRE(i.getName() == "input");
        REQUIRE(i.getNumberOfTrainableParameters() == 0);
    }
    SECTION("Dense")
    {
        auto d =
            ml::Dense<TestType>(10, ml::Activation::Relu, false, ml::Initializer::GlorotUniform,
                                ml::Initializer::Zeros, "dense");

        // Check number of units
        REQUIRE(d.getNumberOfUnits() == 10);

        // Check activation
        REQUIRE(d.getActivation() == ml::Activation::Relu);

        // Check if we are using a bias
        REQUIRE(d.useBias() == false);

        // Check initializers
        REQUIRE(d.getKernelInitializer() == ml::Initializer::GlorotUniform);
        REQUIRE(d.getBiasInitializer() == ml::Initializer::Zeros);

        // Check the name
        REQUIRE(d.getName() == "dense");

        // Check input-descriptor
        IndexVector_t validDims{{1}};
        VolumeDescriptor validDesc(validDims);
        d.setInputDescriptor(validDesc);
        REQUIRE(d.getInputDescriptor() == validDesc);

        d.computeOutputDescriptor();
        IndexVector_t outDims{{10}};
        VolumeDescriptor outDesc(outDims);
        REQUIRE(d.getOutputDescriptor() == outDesc);

        // Note here that we don't use a bias
        REQUIRE(d.getNumberOfTrainableParameters() == 1 * d.getNumberOfUnits());

        // Dense requires a 1D input
        IndexVector_t invalidDims{{1, 10}};
        VolumeDescriptor invalidDesc(invalidDims);
        REQUIRE_THROWS_AS(d.setInputDescriptor(invalidDesc), std::invalid_argument);
    }
}
TEST_SUITE_END();
