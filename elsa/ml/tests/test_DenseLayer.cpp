#include <catch2/catch.hpp>
#include <random>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DenseLayer.h"

using namespace elsa;

struct DenseReference {
    static Eigen::VectorXf forwardPropagate(const Eigen::VectorXf& input,
                                            const Eigen::MatrixXf& weights,
                                            const Eigen::VectorXf& bias)
    {
        return weights * input + bias;
    }

    static Eigen::VectorXf backwardPropagate(const Eigen::VectorXf& input,
                                             const Eigen::MatrixXf& weights,
                                             Eigen::VectorXf& outputGradient)
    {
        Eigen::VectorXf ret;
        ret.resizeLike(input);

        for (int i = 0; i < ret.size(); ++i)
            ret(i) = weights.col(i).dot(outputGradient);

        return ret;
    }
};

TEST_CASE("DenseLayer semantics")
{
    SECTION("Forward Test 1")
    {
        IndexVector_t inputVec(2);
        // One batch dimension and input size is 5
        inputVec << 2, 5;
        DataDescriptor inputDesc(inputVec);
        Eigen::VectorXf inputValues(2 * 5);
        inputValues << 1, 2, 1, 1, 5, 1, 2, 1, 1, 5;
        DataContainer<float> input(inputDesc, inputValues);

        IndexVector_t weightsVec(2);
        // Three neurons and input size is 5
        weightsVec << 3, 5;
        DataDescriptor weightsDesc(weightsVec);
        Eigen::VectorXf weightsValues(3 * 5);
        // clang-format off
        weightsValues <<  1,  2,  3,  4,  5,
                         -1, -2, -3, -4, -5,
                          5,  4,  3,  2,  1;
        // clang-format on
        DataContainer<float> weights(weightsDesc, weightsValues);

        IndexVector_t biasVec(1);
        biasVec << 3;
        DataDescriptor biasDesc(biasVec);
        Eigen::VectorXf biasValues(1 * 3);
        biasValues << 1, 0, 1;
        DataContainer<float> bias(biasDesc, biasValues);

        DenseLayer<float> dense(inputDesc, 3);

        auto backend = dense.getBackend();
        backend->initialize();
        backend->setInput(input);
        backend->compile();
        std::static_pointer_cast<typename DenseLayer<float>::BackendLayerType>(backend)->setWeights(
            weights);
        std::static_pointer_cast<typename DenseLayer<float>::BackendLayerType>(backend)->setBias(
            bias);

        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        auto output = backend->getOutput();

        Eigen::VectorXf required(2 * 3);
        required << 38, -37, 24, 38, -37, 24;

        // We expect one batch and one spatial dimension
        REQUIRE(output.getDataDescriptor().getNumberOfDimensions() == 2);
        REQUIRE(output.getDataDescriptor().getNumberOfCoefficients() == 2 * 3);

        for (int i = 0; i < 2 * 3; ++i)
            REQUIRE(output[i] == required[i]);
    }

    SECTION("Forward Test 2")
    {
        IndexVector_t inputVec(2);
        inputVec << 1, 100;
        DataDescriptor inputDesc(inputVec);
        Eigen::VectorXf inputValues = Eigen::VectorXf::Random(100);
        DataContainer<float> input(inputDesc, inputValues);

        IndexVector_t biasVec(1);
        biasVec << 64;
        DataDescriptor biasDesc(biasVec);
        Eigen::VectorXf biasValues = Eigen::VectorXf::Random(64);
        DataContainer<float> bias(biasDesc, biasValues);

        IndexVector_t weightsVec(2);
        weightsVec << 64, 100;
        DataDescriptor weightsDesc(weightsVec);
        Eigen::MatrixXf weightsMat = Eigen::MatrixXf::Random(64, 100);
        auto required = DenseReference::forwardPropagate(inputValues, weightsMat, biasValues);
        weightsMat.transposeInPlace();
        Eigen::VectorXf weightsValues(Eigen::Map<Eigen::VectorXf>(weightsMat.data(), 64 * 100));

        DataContainer<float> weights(weightsDesc, weightsValues);

        DenseLayer<float> dense(inputDesc, 64);

        auto backend = dense.getBackend();
        backend->initialize();
        backend->setInput(input);
        backend->compile();
        std::static_pointer_cast<typename DenseLayer<float>::BackendLayerType>(backend)->setWeights(
            weights);
        std::static_pointer_cast<typename DenseLayer<float>::BackendLayerType>(backend)->setBias(
            bias);

        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        auto output = backend->getOutput();

        REQUIRE(output.getDataDescriptor() == dense.getOutputDescriptor());

        for (int i = 0; i < output.getDataDescriptor().getNumberOfCoefficients(); ++i)
            REQUIRE(required(i) == Approx(output[i]));
    }

    SECTION("Backward Test 1")
    {
        IndexVector_t gradientVec(2);
        gradientVec << 1, 3;
        DataDescriptor gradientDesc(gradientVec);
        Eigen::VectorXf gradientValues(1 * 3);
        gradientValues << 38, -37, 24;
        DataContainer<float> outputGradient(gradientDesc, gradientValues);

        IndexVector_t weightsVec(2);
        weightsVec << 3, 5;
        DataDescriptor weightsDesc(weightsVec);
        Eigen::VectorXf weightsValues(3 * 5);
        // clang-format off
        weightsValues <<  1,  2,  3,  4,  5,
                         -1, -2, -3, -4, -5,
                          5,  4,  3,  2,  1;
        // clang-format on
        DataContainer<float> weights(weightsDesc, weightsValues);

        IndexVector_t biasVec(1);
        biasVec << 3;
        DataDescriptor biasDesc(biasVec);
        Eigen::VectorXf biasValues(1 * 3);
        biasValues << 1, 0, 1;
        DataContainer<float> bias(biasDesc, biasValues);

        IndexVector_t inputVec(2);
        inputVec << 1, 5;
        Eigen::VectorXf inputValues(1 * 5);
        inputValues << 0, 1, 2, 3, 4;
        DataDescriptor inputDesc(inputVec);
        DataContainer<float> input(inputDesc, inputValues);
        DenseLayer<float> dense(inputDesc, 3);

        auto backend = dense.getBackend();
        backend->initialize();
        backend->setInput(input);
        backend->setOutputGradient(outputGradient);
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);

        backend->compile(PropagationKind::Full);
        std::static_pointer_cast<typename decltype(dense)::BackendLayerType>(backend)->setWeights(
            weights);
        std::static_pointer_cast<typename decltype(dense)::BackendLayerType>(backend)->setBias(
            bias);
        backend->forwardPropagate(s);
        backend->backwardPropagate(s);
        auto inputGradient = backend->getInputGradient();
        auto weightsGradient =
            std::static_pointer_cast<typename decltype(dense)::BackendLayerType>(backend)
                ->getGradientWeights();
        auto biasGradient =
            std::static_pointer_cast<typename decltype(dense)::BackendLayerType>(backend)
                ->getGradientBias();

        Eigen::VectorXd requiredInputGradient(1 * 5);
        requiredInputGradient << 195, 246, 297, 348, 399;

        for (int i = 0; i < requiredInputGradient.size(); ++i)
            REQUIRE(requiredInputGradient(i) == Approx(inputGradient[i]));

        Eigen::VectorXd requiredWeightsGradient(3 * 5);
        requiredWeightsGradient << 0, 38, 76, 114, 152, 0, -37, -74, -111, -148, 0, 24, 48, 72, 96;

        for (int i = 0; i < requiredWeightsGradient.size(); ++i)
            REQUIRE(requiredWeightsGradient(i) == Approx(weightsGradient[i]));

        for (int i = 0; i < 3; ++i)
            REQUIRE(biasGradient[i] == Approx(outputGradient[i]));

    }
}