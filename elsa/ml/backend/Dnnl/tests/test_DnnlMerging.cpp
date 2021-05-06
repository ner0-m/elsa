/**
 * @file test_common.cpp
 *
 * @brief Tests for common ml functionality
 *
 * @author David Tellenbach
 */

#include "doctest/doctest.h"
#include <iostream>

#include "DataContainer.h"
#include "VolumeDescriptor.h"
#include "DnnlMerging.h"

using namespace elsa;
using namespace elsa::ml;
using namespace elsa::ml::detail;
using namespace doctest;

TEST_SUITE_BEGIN("ml-dnnl");

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_CASE("DnnlSum")
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
    DnnlSum<real_t> sum({nchw_desc, nchw_desc, nchw_desc}, nchw_desc);

    REQUIRE(sum.canMerge() == true);
    REQUIRE(sum.isTrainable() == false);
    REQUIRE(sum.needsForwardSynchronisation() == true);

    sum.setInput(dc0, 0);
    sum.setInput(dc1, 1);
    sum.setInput(dc2, 2);

    REQUIRE(sum.getNumberOfInputs() == 3);

    auto engine = sum.getEngine();
    dnnl::stream s(*engine);

    SECTION("Forward propagate")
    {
        sum.compile(PropagationKind::Forward);

        sum.forwardPropagate(s);

        auto output = sum.getOutput();
        REQUIRE(output.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0] == W);
        REQUIRE(output.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1] == H);
        REQUIRE(output.getDataDescriptor().getNumberOfCoefficientsPerDimension()[2] == C);
        REQUIRE(output.getDataDescriptor().getNumberOfCoefficientsPerDimension()[3] == N);

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        REQUIRE(output(w, h, c, n)
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

        sum.compile(PropagationKind::Backward);
        sum.backwardPropagate(s);

        auto inputGradient0 = sum.getInputGradient(0);
        auto inputGradient1 = sum.getInputGradient(1);
        auto inputGradient2 = sum.getInputGradient(2);

        for (int i = 0; i < inputGradient0.getSize(); ++i) {
            REQUIRE(inputGradient0[i] == dc0[i] * outputGradient[i]);
        }

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        REQUIRE(inputGradient0(w, h, c, n)
                                == dc0(w, h, c, n) * outputGradient(w, h, c, n));
                        REQUIRE(inputGradient1(w, h, c, n)
                                == dc1(w, h, c, n) * outputGradient(w, h, c, n));
                        REQUIRE(inputGradient2(w, h, c, n)
                                == dc2(w, h, c, n) * outputGradient(w, h, c, n));
                    }
                }
            }
        }
    }
}
TEST_SUITE_END();
