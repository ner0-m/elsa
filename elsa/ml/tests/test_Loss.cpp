#include <catch2/catch.hpp>
#include "DataContainer.h"
#include "VolumeDescriptor.h"
#include "Loss.h"

using namespace elsa;

TEMPLATE_TEST_CASE("BinaryCrossentropy", "[ml]", float)
{
    IndexVector_t dims{{2, 4}};
    VolumeDescriptor dd(dims);

    // predictions
    Eigen::VectorX<TestType> data_x{{0.6f, 0.4f, 0.4f, 0.6f, 1.f, 0.f, 0.3f, 0.7f}};
    DataContainer<TestType> x(dd, data_x);

    // labels
    Eigen::VectorX<TestType> data_y{{0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 1.f}};
    DataContainer<TestType> y(dd, data_y);

    SECTION("Unweighted SumOverBatchSize")
    {
        auto bce = ml::BinaryCrossentropy(ml::LossReduction::SumOverBatchSize);
        REQUIRE(bce(x, y) == Approx(0.602543f));
    }

    SECTION("Unweighted Sum")
    {
        auto bce = ml::BinaryCrossentropy(ml::LossReduction::Sum);
        REQUIRE(bce(x, y) == Approx(2.410172f));
    }
    SECTION("Gradient Sum")
    {
        auto bce = ml::BinaryCrossentropy(ml::LossReduction::Sum);

        Eigen::VectorXf ref_gradient{{-1.f / 0.8f, -1.f / 0.8f, -1.f / 1.2f, -1.f / 0.8f,
                                      -1.f / 2.f, -1.f / 2.f, -1.f / 0.6f, -1.f / 1.4f}};
        auto gradient = bce.getLossGradient(x, y);

        for (int i = 0; i < gradient.getSize(); ++i)
            REQUIRE(gradient[i] == Approx(ref_gradient[i]));
    }

    SECTION("Gradient SumOverBatchSize")
    {
        auto bce = ml::BinaryCrossentropy(ml::LossReduction::SumOverBatchSize);

        Eigen::VectorXf ref_gradient{{-1.f / 3.2f, -1.f / 3.2f, -1.f / 4.8f, -1.f / 3.2f,
                                      -1.f / 8.f, -1.f / 8.f, -1.f / 2.4f, -1.f / 5.6f}};
        auto gradient = bce.getLossGradient(x, y);

        for (int i = 0; i < gradient.getSize(); ++i)
            REQUIRE(gradient[i] == Approx(ref_gradient[i]));
    }
}

TEMPLATE_TEST_CASE("CategoricalCrossentropy", "[ml]", float)
{
    IndexVector_t dims{{3, 4}};
    VolumeDescriptor dd(dims);

    // predictions
    // clang-format off
    Eigen::VectorX<TestType> data_x{{
      // first batch
      0.05f, 0.95f, 0.f,
      // second batch
      0.1f, 0.8f, 0.1f,
      // third batch
      0.2f, 0.3f, 0.5f,
      // fourth batch
      0.0f, 0.2f, 0.8f}};
    // clang-format on
    DataContainer<TestType> x(dd, data_x);

    // labels in one-hot encoding
    // clang-format off
    Eigen::VectorX<TestType> data_y{{
      // label 1: 1
      0.f, 1.f, 0.f,
      // label 2: 2
      0.f, 0.f, 1.f,
      // label 3: 0
      1.f, 0.f, 0.f, 
      // label 4: 1
      0.f, 1.f, 0.f}};
    // clang-format on

    DataContainer<TestType> y(dd, data_y);

    SECTION("Unweighted SumOverBatchSize")
    {
        auto cce = ml::CategoricalCrossentropy(ml::LossReduction::SumOverBatchSize);
        REQUIRE(cce(x, y) == Approx(1.3931886f));
    }

    SECTION("Unweighted Sum")
    {
        auto cce = ml::CategoricalCrossentropy(ml::LossReduction::Sum);
        REQUIRE(cce(x, y) == Approx(5.5727544f));
    }

    SECTION("Gradient Sum")
    {
        auto cce = ml::CategoricalCrossentropy(ml::LossReduction::Sum);

        Eigen::VectorXf ref_gradient{{0, -1.f / .95f, 0, 0, 0, -10.f, -5.f, 0, 0, 0, -5.f, 0}};
        auto gradient = cce.getLossGradient(x, y);

        for (int i = 0; i < gradient.getSize(); ++i)
            REQUIRE(gradient[i] == Approx(ref_gradient[i]));
    }

    SECTION("Gradient SumOverBatchSize")
    {
        auto cce = ml::CategoricalCrossentropy(ml::LossReduction::SumOverBatchSize);

        Eigen::VectorXf ref_gradient{
            {0, -1.f / 3.8f, 0, 0, 0, -10.f / 4.f, -5.f / 4.f, 0, 0, 0, -5.f / 4.f, 0}};
        auto gradient = cce.getLossGradient(x, y);

        for (int i = 0; i < gradient.getSize(); ++i)
            REQUIRE(gradient[i] == Approx(ref_gradient[i]));
    }
}

TEMPLATE_TEST_CASE("SparseCategoricalCrossentropy", "[ml]", float)
{
    IndexVector_t predictionDims{{3, 4}};
    VolumeDescriptor predictionDesc(predictionDims);

    // predictions
    Eigen::VectorX<TestType> data_x{
        {0.05f, 0.95f, 0.f, 0.1f, 0.8f, 0.1f, 0.2f, 0.3f, 0.5f, 0.0f, 0.2f, 0.8f}};
    DataContainer<TestType> x(predictionDesc, data_x);

    IndexVector_t labelDims{{4}};
    VolumeDescriptor labelDesc(labelDims);

    // labels
    Eigen::VectorX<TestType> data_y{{1.f, 2.f, 0.f, 1.f}};
    DataContainer<TestType> y(labelDesc, data_y);

    SECTION("Unweighted SumOverBatchSize")
    {
        auto scce = ml::SparseCategoricalCrossentropy(ml::LossReduction::SumOverBatchSize);
        REQUIRE(scce(x, y) == Approx(1.3931886f));
    }

    SECTION("Unweighted Sum")
    {
        auto scce = ml::SparseCategoricalCrossentropy(ml::LossReduction::Sum);
        REQUIRE(scce(x, y) == Approx(5.5727544f));
    }
    SECTION("Gradient Sum")
    {
        auto scce = ml::SparseCategoricalCrossentropy(ml::LossReduction::Sum);

        Eigen::VectorXf ref_gradient{{0, -1.f / .95f, 0, 0, 0, -10.f, -5.f, 0, 0, 0, -5.f, 0}};
        auto gradient = scce.getLossGradient(x, y);

        for (int i = 0; i < gradient.getSize(); ++i)
            REQUIRE(gradient[i] == Approx(ref_gradient[i]));
    }

    SECTION("Gradient SumOverBatchSize")
    {
        auto scce = ml::SparseCategoricalCrossentropy(ml::LossReduction::SumOverBatchSize);

        Eigen::VectorXf ref_gradient{
            {0, -1.f / 3.8f, 0, 0, 0, -10.f / 4.f, -5.f / 4.f, 0, 0, 0, -5.f / 4.f, 0}};
        auto gradient = scce.getLossGradient(x, y);

        for (int i = 0; i < gradient.getSize(); ++i)
            REQUIRE(gradient[i] == Approx(ref_gradient[i]));
    }
}

TEMPLATE_TEST_CASE("MeanSquaredError", "[ml]", float)
{
    IndexVector_t dims{{3, 2}};
    VolumeDescriptor dd(dims);

    // predictions
    Eigen::VectorX<TestType> data_x{{1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f}};
    DataContainer<TestType> x(dd, data_x);

    // labels
    Eigen::VectorX<TestType> data_y{{0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f}};
    DataContainer<TestType> y(dd, data_y);

    SECTION("Unweighted SumOverBatchSize")
    {
        auto mse = ml::MeanSquaredError(ml::LossReduction::SumOverBatchSize);
        REQUIRE(mse(x, y) == Approx(0.33333334f));
    }

    SECTION("Unweighted Sum")
    {
        auto mse = ml::MeanSquaredError(ml::LossReduction::Sum);
        REQUIRE(mse(x, y) == Approx(0.6666667f));
    }

    SECTION("Gradient Sum")
    {
        auto mse = ml::MeanSquaredError(ml::LossReduction::Sum);
        Eigen::VectorXf refDerivative{{-2.f / 3.f, 0, 0, -2.f / 3.f, 0, 0}};
        auto derivative = mse.getLossGradient(x, y);

        for (int i = 0; i < derivative.getSize(); ++i)
            REQUIRE(derivative[i] == Approx(refDerivative[i]));
    }

    SECTION("Gradient SumOverBatchSize")
    {
        auto mse = ml::MeanSquaredError(ml::LossReduction::SumOverBatchSize);
        Eigen::VectorXf refDerivative{{-2.f / 6.f, 0, 0, -2.f / 6.f, 0, 0}};
        auto derivative = mse.getLossGradient(x, y);

        for (int i = 0; i < derivative.getSize(); ++i)
            REQUIRE(derivative[i] == Approx(refDerivative[i]));
    }
}
