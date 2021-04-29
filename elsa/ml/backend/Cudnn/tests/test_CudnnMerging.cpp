#include <catch2/catch.hpp>
#include <random>

#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "CudnnMerging.h"
#include "CudnnDataContainerInterface.h"

using namespace elsa;
using namespace elsa::ml::detail;

TEST_CASE("CudnnSum", "[ml][cudnn]")
{
    index_t N = 11;
    index_t C = 22;
    index_t H = 33;
    index_t W = 44;

    IndexVector_t dims{{W, H, C, N}};
    VolumeDescriptor desc(dims);

    Eigen::VectorXf data(desc.getNumberOfCoefficients());

    data.setRandom();
    DataContainer<real_t> dc0(desc, data);

    data.setRandom();
    DataContainer<real_t> dc1(desc, data);

    data.setRandom();
    DataContainer<real_t> dc2(desc, data);

    IndexVector_t nchw_dims{{N, C, H, W}};
    VolumeDescriptor nchw_desc(nchw_dims);
    CudnnSum<real_t> sum({nchw_desc, nchw_desc, nchw_desc}, nchw_desc);

    REQUIRE(sum.canMerge() == true);
    REQUIRE(sum.isTrainable() == false);
    REQUIRE(sum.needsForwardSynchronisation() == true);

    sum.setInput(dc0, 0);
    sum.setInput(dc1, 1);
    sum.setInput(dc2, 2);

    REQUIRE(sum.getNumberOfInputs() == 3);

    SECTION("Forward propagate")
    {
        sum.forwardPropagate();

        auto output = sum.getOutput();
        REQUIRE(output.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0] == N);
        REQUIRE(output.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1] == C);
        REQUIRE(output.getDataDescriptor().getNumberOfCoefficientsPerDimension()[2] == H);
        REQUIRE(output.getDataDescriptor().getNumberOfCoefficientsPerDimension()[3] == W);

        for (int i = 0; i < output.getSize(); ++i) {
            REQUIRE(output[i] == dc0[i] + dc1[i] + dc2[i]);
        }

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        REQUIRE(output.viewAs(desc)(w, h, c, n)
                                == dc0(w, h, c, n) + dc1(w, h, c, n) + dc2(w, h, c, n));
                    }
                }
            }
        }
    }

    SECTION("Backward propagate")
    {
        Eigen::VectorXf outputGradientData(sum.getOutputDescriptor().getNumberOfCoefficients());
        outputGradientData.setRandom();
        DataContainer<real_t> outputGradient(desc, outputGradientData);
        sum.setOutputGradient(outputGradient);

        sum.backwardPropagate();

        auto inputGradient0 = sum.getInputGradient(0);
        auto inputGradient1 = sum.getInputGradient(1);
        auto inputGradient2 = sum.getInputGradient(2);

        for (int i = 0; i < inputGradient0.getSize(); ++i) {
            REQUIRE(inputGradient0[i] == dc0[i] * outputGradient[i]);
            REQUIRE(inputGradient1[i] == dc1[i] * outputGradient[i]);
            REQUIRE(inputGradient2[i] == dc2[i] * outputGradient[i]);
        }

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        REQUIRE(inputGradient0.viewAs(desc)(w, h, c, n)
                                == dc0(w, h, c, n) * outputGradient(w, h, c, n));
                        REQUIRE(inputGradient1.viewAs(desc)(w, h, c, n)
                                == dc1(w, h, c, n) * outputGradient(w, h, c, n));
                        REQUIRE(inputGradient2.viewAs(desc)(w, h, c, n)
                                == dc2(w, h, c, n) * outputGradient(w, h, c, n));
                    }
                }
            }
        }
    }
}