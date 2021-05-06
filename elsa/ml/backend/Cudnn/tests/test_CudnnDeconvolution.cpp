#include "doctest/doctest.h"
#include <random>

#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "CudnnConvolution.h"
#include "CudnnDataContainerInterface.h"

using namespace elsa;
using namespace elsa::ml;
using namespace elsa::ml::detail;
using namespace doctest;

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_SUITE_BEGIN("ml-cudnn");

TEST_CASE("CudnnDeconvolution")
{
    SECTION("Test 1")
    {
        index_t N = 1;
        index_t C = 1;
        index_t H = 2;
        index_t W = 2;

        IndexVector_t inputDims{{N, C, H, W}};
        VolumeDescriptor inputDesc(inputDims);

        IndexVector_t kernelDims{{N, C, H, W}};
        VolumeDescriptor kernelDesc(inputDims);

        IndexVector_t outputDims{{N, C, 3, 3}};
        VolumeDescriptor outputDesc(outputDims);

        auto layer = CudnnDeconvolution<real_t>(
            inputDesc, outputDesc, kernelDesc, IndexVector_t({{1, 1}}), IndexVector_t({{0}}),
            IndexVector_t({{0}}), true, Initializer::Ones, Initializer::Ones);

        Eigen::VectorXf inputData{{0, 1, 2, 3}};
        DataContainer<real_t> dc(inputDesc, inputData);

        layer.setInput(dc);
        layer.compileForwardStream();
        layer.forwardPropagate();

        auto output = layer.getOutput();

        Eigen::VectorXf refData{{0 + 1, 1 + 1, 1 + 1, 2 + 1, 6 + 1, 4 + 1, 2 + 1, 5 + 1, 3 + 1}};
        DataContainer<real_t> refDc(layer.getOutputDescriptor(), refData);
        // Eigen::VectorXf requiredOutput  {{}}
        for (int i = 0; i < output.getSize(); ++i)
            REQUIRE(output[i] == refDc[i]);

        layer.setNumberOfOutputGradients(1);
        layer.setOutputGradient(output);
        layer.compileBackwardStream();
        layer.backwardPropagate();

        auto inGrad = layer.getInputGradient();

        for (int i = 0; i < inGrad.getSize(); ++i)
            std::cout << inGrad[i] << "\n";
    }
}

TEST_SUITE_END();
