/**
 * \file test_DnnlConvolution.cpp
 *
 * \brief Tests for DnnlConvolution
 *
 * \author David Tellenbach
 */

#include <catch2/catch.hpp>
#include <iostream>

#include "DataContainer.h"
#include "VolumeDescriptor.h"
#include "DnnlConvolution.h"

using namespace elsa;
using namespace elsa::ml;
using namespace elsa::ml::detail;

TEST_CASE("DnnlConvolution", "[ml][dnnl]")
{
    // Example taken from http://cs231n.github.io/convolutional-networks/
    SECTION("Test 1")
    {
        IndexVector_t nchw_inputDims(4);
        inputVec << 1, 3, 5, 5;
        VolumeDescriptor nchw_inputDesc(nchw_inputDims);

        IndexVector_t whcn_inputDims(4);
        inputVec << 5, 5, 3, 1;
        VolumeDescriptor whcn_inputDesc(whcn_inputDims);

        IndexVector_t weightsVec(4);
        weightsVec << 2, 3, 3, 3;
        VolumeDescriptor weightsDesc(weightsVec);

        IndexVector_t stridesVec(2);
        stridesVec << 2, 2;

        IndexVector_t paddingVec(2);
        paddingVec << 1, 1;

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

        Eigen::VectorXf vec3(2);
        vec3 << 1, 0;
        IndexVector_t biasVec(1);
        biasVec << 2;
        VolumeDescriptor biasDesc(biasVec);
        DataContainer<float> bias(biasDesc, vec3);

        DnnlConvolution<real_t> conv(nchw_inputDesc, )
    }
}