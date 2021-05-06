#include "doctest/doctest.h"

#include "testHelpers.h"
#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "CudnnReshape.h"
#include "CudnnDataContainerInterface.h"

using namespace elsa;
using namespace elsa::ml;
using namespace elsa::ml::detail;
using namespace doctest;

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_SUITE_BEGIN("ml-cudnn");

TEST_CASE("CudnnFlatten")
{
    IndexVector_t inputDimensions{{11, 22, 33, 44}};
    VolumeDescriptor inputDescriptor(inputDimensions);

    auto input =
        std::get<0>(generateRandomContainer<real_t>(inputDescriptor, DataHandlerType::CPU));

    IndexVector_t outputDimensions{{11, 22 * 33 * 44}};
    VolumeDescriptor outputDescriptor(outputDimensions);

    auto layer = CudnnFlatten<real_t>(inputDescriptor);
    layer.compileForwardStream();
    layer.setInput(input);

    auto output = layer.getOutput();

    REQUIRE(isApprox(input, output));
}

TEST_CASE("CudnnUpsample")
{
    const int H = 3;
    const int W = 3;

    // nchw
    IndexVector_t dims{{2, 3, H, W}};
    VolumeDescriptor desc(dims);

    const int h = 2;
    const int w = 2;

    IndexVector_t outDims{{2, 3, H * h, W * w}};
    VolumeDescriptor outDesc(outDims);

    auto layer = CudnnUpsampling<real_t>(desc, outDesc, Interpolation::NearestNeighbour);

    Eigen::VectorXf vec = Eigen::VectorXf::LinSpaced(dims.prod(), 0, dims.prod() - 1);

    DataContainer<real_t> data(desc, vec);
    layer.setInput(data);

    layer.compileForwardStream();
    layer.forwardPropagate();

    auto output = layer.getOutput();

    Eigen::VectorXf requiredData(outDesc.getNumberOfCoefficients());

    // clang-format off
    requiredData <<
    0, 0, 1, 1, 2, 2, 
    0, 0, 1, 1, 2, 2, 
    3, 3, 4, 4, 5, 5, 
    3, 3, 4, 4, 5, 5, 
    6, 6, 7, 7, 8, 8,
    6, 6, 7, 7, 8, 8,
    
     9,  9, 10, 10, 11, 11,
     9,  9, 10, 10, 11, 11,
    12, 12, 13, 13, 14, 14,
    12, 12, 13, 13, 14, 14,
    15, 15, 16, 16, 17, 17,
    15, 15, 16, 16, 17, 17,

    18, 18, 19, 19, 20, 20,
    18, 18, 19, 19, 20, 20,
    21, 21, 22, 22, 23, 23,
    21, 21, 22, 22, 23, 23,
    24, 24, 25, 25, 26, 26,
    24, 24, 25, 25, 26, 26,

    27, 27, 28, 28, 29, 29,
    27, 27, 28, 28, 29, 29,
    30, 30, 31, 31, 32, 32,
    30, 30, 31, 31, 32, 32,
    33, 33, 34, 34, 35, 35,
    33, 33, 34, 34, 35, 35,

    36, 36, 37, 37, 38, 38,
    36, 36, 37, 37, 38, 38,
    39, 39, 40, 40, 41, 41,
    39, 39, 40, 40, 41, 41,
    42, 42, 43, 43, 44, 44,
    42, 42, 43, 43, 44, 44,

    45, 45, 46, 46, 47, 47,
    45, 45, 46, 46, 47, 47,
    48, 48, 49, 49, 50, 50,
    48, 48, 49, 49, 50, 50,
    51, 51, 52, 52, 53, 53,
    51, 51, 52, 52, 53, 53;
    // clang-format on

    for (int i = 0; i < output.getSize(); ++i) {
        REQUIRE(output[i] == Approx(requiredData[i]));
    }

    layer.compileBackwardStream();
    layer.setNumberOfOutputGradients(1);
    layer.setOutputGradient(output);
    layer.backwardPropagate();
    auto ingrad = layer.getInputGradient();

    REQUIRE(ingrad.getDataDescriptor() == desc);

    for (int i = 0; i < ingrad.getSize(); ++i) {
        REQUIRE(ingrad[i] == Approx(data[i]));
    }
}

TEST_SUITE_END();
