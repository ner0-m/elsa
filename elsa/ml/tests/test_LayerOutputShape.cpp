#include <catch2/catch.hpp>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "ActivationLayer.h"
#include "ConvLayer.h"
#include "PoolingLayer.h"
#include "DenseLayer.h"
#include "SoftmaxLayer.h"

using namespace elsa;

TEMPLATE_TEST_CASE("ActivationLayerOutputShapes", "elsa_ml", (Abs<float, MlBackend::Dnnl>),
                   (BoundedRelu<float, MlBackend::Dnnl>), (Elu<float, MlBackend::Dnnl>),
                   (Exp<float, MlBackend::Dnnl>), (Gelu<float, MlBackend::Dnnl>),
                   (Linear<float, MlBackend::Dnnl>), (Logistic<float, MlBackend::Dnnl>),
                   (Relu<float, MlBackend::Dnnl>), (SoftRelu<float, MlBackend::Dnnl>),
                   (Sqrt<float, MlBackend::Dnnl>), (Swish<float, MlBackend::Dnnl>),
                   (Tanh<float, MlBackend::Dnnl>) )
{
    IndexVector_t vec(4);
    vec << 1, 2, 3, 4;
    DataDescriptor inputDesc(vec);
    // Activations don't alter the input shape
    TestType layer(inputDesc);
    REQUIRE(layer.getOutputDescriptor() == inputDesc);
}

TEST_CASE("LayerOutputShapes", "elsa_ml")
{
    SECTION("Conv")
    {
        {
            IndexVector_t inputVec(4);
            inputVec << 32, 3, 227, 227;
            DataDescriptor inputDesc(inputVec);

            IndexVector_t weightsVec(4);
            weightsVec << 96, 3, 11, 11;
            DataDescriptor weightsDesc(weightsVec);

            IndexVector_t stridesVec(2);
            stridesVec << 4, 4;

            IndexVector_t paddingVec(2);
            paddingVec << 0, 0;

            ConvLayer<float> conv(inputDesc, weightsDesc, stridesVec, paddingVec);

            IndexVector_t outputVec(4);
            outputVec << 32, 96, 55, 55;
            DataDescriptor outputDesc(outputVec);
            REQUIRE((conv.getOutputDescriptor() == outputDesc));
        }
    }

    SECTION("Pooling")
    {
        {
            IndexVector_t inputVec(4);
            inputVec << 32, 96, 55, 55;
            DataDescriptor inputDesc(inputVec);

            IndexVector_t stridesVec(2);
            stridesVec << 2, 2;

            IndexVector_t poolingVec(2);
            poolingVec << 3, 3;

            PoolingLayer<float> pool(inputDesc, poolingVec, stridesVec);

            IndexVector_t outputVec(4);
            outputVec << 32, 96, 27, 27;
            DataDescriptor outputDesc(outputVec);
            REQUIRE(pool.getOutputDescriptor() == outputDesc);
        }
    }

    SECTION("Dense")
    {
        {
            IndexVector_t inputVec(4);
            inputVec << 32, 256, 6, 6;
            DataDescriptor inputDesc(inputVec);

            DenseLayer<float> dense(inputDesc, 4096);

            IndexVector_t outputVec(2);
            outputVec << 32, 4096;
            DataDescriptor outputDesc(outputVec);

            REQUIRE(dense.getOutputDescriptor() == outputDesc);
        }
        {
            IndexVector_t inputVec(2);
            inputVec << 32, 4096;
            DataDescriptor inputDesc(inputVec);

            DenseLayer<float> dense(inputDesc, 1000);

            IndexVector_t outputVec(2);
            outputVec << 32, 1000;
            DataDescriptor outputDesc(outputVec);

            REQUIRE(dense.getOutputDescriptor() == outputDesc);
        }
    }

    SECTION("Softmax")
    {
        {
            IndexVector_t vec(4);
            vec << 1, 2, 3, 4;
            DataDescriptor inputDesc(vec);
            SoftmaxLayer layer(inputDesc);
            REQUIRE(layer.getOutputDescriptor() == inputDesc);
        }
    }
}
