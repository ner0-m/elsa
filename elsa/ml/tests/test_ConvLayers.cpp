#include <catch2/catch.hpp>
#include "Input.h"
#include "Conv.h"

using namespace elsa;

TEMPLATE_TEST_CASE("ConvShapes", "[ml]", float)
{
    SECTION("Conv1D")
    {
        // ch
        IndexVector_t inputDims{{50}};
        VolumeDescriptor inputDesc(inputDims);
        auto input = ml::Input(inputDesc, 1);

        IndexVector_t filterDims{{3}};
        VolumeDescriptor filterDesc(filterDims);

        auto conv =
            ml::Conv1D<TestType>(5, filterDesc, ml::Activation::Relu, 2, ml::Padding::Valid, false);

        REQUIRE(conv.getNumberOfFilters() == 5);
        REQUIRE(conv.getFilterDescriptor() == filterDesc);
        REQUIRE(conv.getActivation() == ml::Activation::Relu);
        REQUIRE(conv.getStrides() == 2);
        REQUIRE(conv.useBias() == false);
    }

    SECTION("Conv2D")
    {
        IndexVector_t inputDims{{255, 255, 64}};
        VolumeDescriptor inputDesc(inputDims);

        auto input = ml::Input(inputDesc, 1);

        auto layer = ml::Conv2D<real_t>(3, std::array<index_t, 3>({7, 7, 64}), ml::Activation::Relu,
                                        3, ml::Padding::Same);
        layer.setInput(&input);

        layer.setInputDescriptor(inputDesc);
        layer.computeOutputDescriptor();

        IndexVector_t outDims{{255, 255, 3}};
        VolumeDescriptor outDesc(outDims);

        REQUIRE(layer.getOutputDescriptor().getNumberOfCoefficientsPerDimension() == outDims);
    }
}
