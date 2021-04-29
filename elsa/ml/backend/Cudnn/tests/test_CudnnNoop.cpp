#include <catch2/catch.hpp>
#include <random>

#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "CudnnNoop.h"
#include "CudnnDataContainerInterface.h"

using namespace elsa;
using namespace elsa::ml::detail;

TEST_CASE("CudnnNoop", "[ml][cudnn]")
{
    index_t N = 11;
    index_t C = 22;
    index_t H = 33;
    index_t W = 44;

    IndexVector_t dims{{W, H, C, N}};
    VolumeDescriptor desc(dims);

    Eigen::VectorXf data(desc.getNumberOfCoefficients());

    data.setRandom();
    DataContainer<real_t> input(desc, data);

    IndexVector_t nchw_dims{{N, C, H, W}};
    VolumeDescriptor nchw_desc(nchw_dims);
    auto layer = CudnnNoop<float>(nchw_desc);

    REQUIRE(layer.getInputDescriptor() == nchw_desc);
    REQUIRE(layer.getOutputDescriptor() == nchw_desc);
    REQUIRE(layer.isTrainable() == false);
    REQUIRE(layer.canMerge() == false);
    REQUIRE(layer.getName() == "CudnnNoop");

    layer.setInput(input);
    layer.forwardPropagate();
    auto output = layer.getOutput();

    for (int i = 0; i < output.getSize(); ++i)
        REQUIRE(output[i] == input[i]);

    layer.setOutputGradient(output);
    layer.backwardPropagate();
    auto inGrad = layer.getInputGradient();

    for (int i = 0; i < output.getSize(); ++i)
        REQUIRE(inGrad[i] == input[i]);
}
