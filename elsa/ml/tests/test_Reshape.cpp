#include "doctest/doctest.h"
#include "Input.h"
#include "Reshape.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("ml");

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_CASE("Reshape")
{
    IndexVector_t inputDims{{3, 4, 5, 2}};
    VolumeDescriptor inputDesc(inputDims);

    VolumeDescriptor targetShape({60, 2});
    auto layer = ml::Reshape(targetShape, "Reshape");
    REQUIRE(layer.getName() == "Reshape");

    layer.setInputDescriptor(inputDesc);
    REQUIRE(layer.getInputDescriptor() == inputDesc);

    layer.computeOutputDescriptor();

    REQUIRE(layer.getOutputDescriptor() == targetShape);
}

TEST_CASE("Flatten")
{
    IndexVector_t inputDims{{3, 4, 5, 2}};

    VolumeDescriptor inputDesc(inputDims);

    auto input = ml::Input(inputDesc);

    auto layer = ml::Flatten("Flatten");
    REQUIRE(layer.getName() == "Flatten");

    layer.setInput(&input);

    input.computeOutputDescriptor();

    REQUIRE(input.getOutputDescriptor() == inputDesc);

    layer.setInputDescriptor(input.getOutputDescriptor());
    layer.computeOutputDescriptor();
    REQUIRE(layer.getInputDescriptor(0) == inputDesc);

    IndexVector_t requiredOutDims{{inputDims[0] * inputDims[1] * inputDims[2] * inputDims[3]}};

    VolumeDescriptor requiredOutDesc(requiredOutDims);
    REQUIRE(layer.getOutputDescriptor() == requiredOutDesc);
}
TEST_SUITE_END();
