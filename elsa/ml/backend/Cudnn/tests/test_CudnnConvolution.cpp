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

TEST_CASE("CudnnConvolution")
{
    SECTION("No padding, Stride 1")
    {
        DataDescriptor inputDesc({1, 3, 5, 5});
        DataDescriptor weightsDesc({2, 3, 3, 3});

        Eigen::VectorXf vec(1 * 3 * 5 * 5);
        // clang-format off
        vec <<  // First channel
            1, 2, 1, 1, 1,
            1, 2, 2, 1, 1,
            2, 0, 2, 0, 0,
            1, 1, 2, 2, 0,
            2, 1, 1, 0, 2,
            // Second channel
            1, 0, 2, 0, 2,
            1, 0, 2, 1, 2,
            0, 1, 1, 1, 1,
            1, 0, 2, 0, 1,
            2, 2, 0, 0, 1,
            // Third channel
            2, 2, 0, 1, 2,
            2, 1, 0, 0, 2,
            1, 2, 0, 1, 0,
            2, 0, 0, 2, 2,
            2, 1, 2, 0, 0;
        // clang-format on
        DataContainer<float> input(inputDesc, vec);

        Eigen::VectorXf vec2(2 * 3 * 3 * 3);
        // clang-format off
        vec2 << // First filter
              // First channel
               0,  0, -1,
              -1,  0, -1,
               1, -1,  0,
              // Second channel
               1, -1, -1,
               1,  1,  1,
              -1,  0,  1,
              // Third channel
               0,  0, -1,
               1, -1, -1,
               0, -1,  0,
              // Second filter
              // First channel
              -1, -1, -1,
              -1, -1,  1,
               0,  0,  1,
              // Second channel
              -1,  0,  1,
              -1,  0,  1,
               1, -1,  1,
              // Third channel
               0,  0,  1,
              -1,  0,  1,
              -1,  0, -1;
        // clang-format on

        DataContainer<float> weights(weightsDesc, vec2);

        Eigen::VectorXf vec3({{1, 0}});
        DataDescriptor biasDesc({2});
        DataContainer<float> bias(biasDesc, vec3);
    }
}

TEST_SUITE_END();
