/**
 * @file test_QuadricProblem.cpp
 *
 * @brief Tests for the QuqricProblem class
 *
 * @author Nikola Dinev
 */
#include <catch2/catch.hpp>
#include "QuadricProblem.h"
#include "Identity.h"
#include "Scaling.h"
#include "L2NormPow2.h"
#include "WeightedL2NormPow2.h"
#include "Huber.h"
#include "Logger.h"
#include "VolumeDescriptor.h"

#include <array>

using namespace elsa;

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEMPLATE_TEST_CASE("Scenario: Testing QuadricProblem", "", QuadricProblem<float>,
                   QuadricProblem<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("a Quadric functional")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 11, 7;
        VolumeDescriptor dd{numCoeff};

        auto scaleFactor = static_cast<data_t>(3.0);
        Scaling<data_t> scalingOp{dd, scaleFactor};

        Eigen::Matrix<data_t, -1, 1> randomData{dd.getNumberOfCoefficients()};
        randomData.setRandom();
        DataContainer<data_t> dc{dd, randomData};

        Quadric<data_t> quad{scalingOp, dc};

        WHEN("constructing a QuadricProblem without x0 from it")
        {
            TestType prob{quad};

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer<data_t> dcZero(dd);
                dcZero = 0;
                REQUIRE(prob.getCurrentSolution() == dcZero);

                REQUIRE(prob.evaluate() == 0);
                REQUIRE(prob.getGradient() == static_cast<data_t>(-1.0) * dc);

                auto hessian = prob.getHessian();
                REQUIRE(hessian == leaf(scalingOp));
            }
        }

        WHEN("constructing a QuadricProblem with x0 from it")
        {
            Eigen::Matrix<data_t, -1, 1> x0Vec{dd.getNumberOfCoefficients()};
            x0Vec.setRandom();
            DataContainer<data_t> x0{dd, x0Vec};

            TestType prob{quad, x0};

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE(prob.getCurrentSolution() == x0);

                REQUIRE(prob.evaluate()
                        == Approx(static_cast<data_t>(0.5 * scaleFactor) * x0.squaredL2Norm()
                                  - x0.dot(dc)));

                auto gradient = prob.getGradient();
                DataContainer gradientDirect = scaleFactor * x0 - dc;
                for (index_t i = 0; i < gradient.getSize(); ++i)
                    REQUIRE(gradient[i] == Approx(gradientDirect[i]));

                auto hessian = prob.getHessian();
                REQUIRE(hessian == leaf(scalingOp));
            }
        }
    }

    GIVEN("a spd operator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 11, 7;
        VolumeDescriptor dd{numCoeff};

        data_t scaleFactor = 3.0;
        Scaling<data_t> scalingOp{dd, scaleFactor};

        Eigen::Matrix<data_t, -1, 1> randomData{dd.getNumberOfCoefficients()};
        randomData.setRandom();
        DataContainer<data_t> dc{dd, randomData};

        WHEN("constructing a QuadricProblem without x0 from it")
        {
            TestType prob{scalingOp, dc, true};

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer<data_t> dcZero(dd);
                dcZero = 0;
                REQUIRE(prob.getCurrentSolution() == dcZero);

                REQUIRE(prob.evaluate() == 0);
                REQUIRE(prob.getGradient() == static_cast<data_t>(-1.0) * dc);

                auto hessian = prob.getHessian();
                REQUIRE(hessian == leaf(scalingOp));
            }
        }

        WHEN("constructing a QuadricProblem with x0 from it")
        {
            Eigen::Matrix<data_t, -1, 1> x0Vec{dd.getNumberOfCoefficients()};
            x0Vec.setRandom();
            DataContainer<data_t> x0{dd, x0Vec};

            TestType prob{scalingOp, dc, x0, true};

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE(prob.getCurrentSolution() == x0);

                REQUIRE(prob.evaluate()
                        == Approx(static_cast<data_t>(0.5 * scaleFactor) * x0.squaredL2Norm()
                                  - x0.dot(dc)));

                auto gradient = prob.getGradient();
                DataContainer gradientDirect = scaleFactor * x0 - dc;
                for (index_t i = 0; i < gradient.getSize(); ++i)
                    REQUIRE(gradient[i] == Approx(gradientDirect[i]));

                auto hessian = prob.getHessian();
                REQUIRE(hessian == leaf(scalingOp));
            }
        }
    }

    GIVEN("a non-spd operator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 11, 7;
        VolumeDescriptor dd{numCoeff};

        data_t scaleFactor = -3.0;
        Scaling<data_t> scalingOp{dd, scaleFactor};

        Eigen::Matrix<data_t, -1, 1> randomData{dd.getNumberOfCoefficients()};
        randomData.setRandom();
        DataContainer<data_t> dc{dd, randomData};

        WHEN("Constructing a QuadricProblem without x0 from it")
        {
            TestType prob{scalingOp, dc, false};

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer<data_t> dcZero(dd);
                dcZero = 0;
                REQUIRE(prob.getCurrentSolution() == dcZero);

                REQUIRE(prob.evaluate() == 0);
                REQUIRE(prob.getGradient() == static_cast<data_t>(-scaleFactor) * dc);

                auto hessian = prob.getHessian();
                REQUIRE(hessian == leaf(adjoint(scalingOp) * scalingOp));
            }
        }

        WHEN("Constructing a QuadricProblem with x0 from it")
        {
            Eigen::Matrix<data_t, -1, 1> x0Vec{dd.getNumberOfCoefficients()};
            x0Vec.setRandom();
            DataContainer<data_t> x0{dd, x0Vec};

            TestType prob{scalingOp, dc, x0, false};

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE(prob.getCurrentSolution() == x0);

                REQUIRE(prob.evaluate()
                        == Approx(static_cast<data_t>(0.5 * scaleFactor * scaleFactor)
                                      * x0.squaredL2Norm()
                                  - scaleFactor * x0.dot(dc)));

                auto gradient = prob.getGradient();
                DataContainer gradientDirect = scaleFactor * (scaleFactor * x0) - scaleFactor * dc;
                for (index_t i = 0; i < gradient.getSize(); ++i)
                    REQUIRE(gradient[i] == Approx(gradientDirect[i]).margin(0.00001));

                auto hessian = prob.getHessian();
                REQUIRE(hessian == leaf(adjoint(scalingOp) * scalingOp));
            }
        }
    }

    GIVEN("an optimization problem")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 11, 7;
        VolumeDescriptor dd{numCoeff};

        data_t scaleFactor = -3.0;
        Scaling<data_t> scalingOp{dd, scaleFactor};

        Eigen::Matrix<data_t, -1, 1> randomData{dd.getNumberOfCoefficients()};
        randomData.setRandom();
        DataContainer<data_t> dc{dd, randomData};

        data_t weightFactor = 2.0;
        Scaling<data_t> weightingOp{dd, weightFactor};

        WHEN("trying to convert a problem with a non-quadric non-wls data term")
        {
            Problem prob{Huber<data_t>{dd}};
            THEN("an exception is thrown") { REQUIRE_THROWS(QuadricProblem{prob}); }
        }

        WHEN("trying to convert a problem with a non-quadric non-wls regularization term")
        {
            Problem prob{L2NormPow2<data_t>{dd},
                         RegularizationTerm{static_cast<data_t>(0.5), Huber<data_t>{dd}}};
            THEN("an exception is thrown") { REQUIRE_THROWS(QuadricProblem{prob}); }
        }

        WHEN("converting an optimization problem that has only a quadric data term")
        {
            randomData.setRandom();
            DataContainer<data_t> x0{dd, randomData};

            Problem<data_t> initialProb{Quadric<data_t>{weightingOp, dc}, x0};

            QuadricProblem<data_t> prob{initialProb};
            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE(prob.getCurrentSolution() == x0);

                REQUIRE(prob.evaluate()
                        == Approx(static_cast<data_t>(0.5 * weightFactor) * x0.squaredL2Norm()
                                  - x0.dot(dc)));
                REQUIRE(prob.getGradient() == (weightFactor * x0) - dc);

                auto hessian = prob.getHessian();
                REQUIRE(hessian == leaf(weightingOp));
            }
        }

        WHEN("calling the conversion constructor on a QadricProblem")
        {
            randomData.setRandom();
            DataContainer<data_t> x0{dd, randomData};

            QuadricProblem<data_t> initialProb{Quadric<data_t>{weightingOp, dc}, x0};

            QuadricProblem<data_t> prob{initialProb};
            THEN("conversion yields the same result as cloning")
            {
                REQUIRE(*initialProb.clone() == prob);
            }

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE(prob.getCurrentSolution() == x0);

                REQUIRE(prob.evaluate()
                        == Approx(static_cast<data_t>(0.5 * weightFactor) * x0.squaredL2Norm()
                                  - x0.dot(dc)));
                REQUIRE(prob.getGradient() == (weightFactor * x0) - dc);

                auto hessian = prob.getHessian();
                REQUIRE(hessian == leaf(weightingOp));
            }
        }

        randomData.setRandom();
        DataContainer<data_t> x0{dd, randomData};

        Scaling A{dd, static_cast<data_t>(2.0)};
        randomData.setRandom();
        DataContainer<data_t> b{dd, randomData};

        std::vector<std::unique_ptr<Functional<data_t>>> dataTerms;
        dataTerms.push_back(std::make_unique<Quadric<data_t>>(dd));
        dataTerms.push_back(std::make_unique<Quadric<data_t>>(A));
        dataTerms.push_back(std::make_unique<Quadric<data_t>>(b));
        dataTerms.push_back(std::make_unique<Quadric<data_t>>(A, b));

        Scaling L{dd, static_cast<data_t>(0.5)};
        randomData.setRandom();
        DataContainer c{dd, randomData};

        std::vector<std::unique_ptr<Quadric<data_t>>> regFunc;
        regFunc.push_back(std::make_unique<Quadric<data_t>>(dd));
        regFunc.push_back(std::make_unique<Quadric<data_t>>(A));
        regFunc.push_back(std::make_unique<Quadric<data_t>>(b));
        regFunc.push_back(std::make_unique<Quadric<data_t>>(A, b));

        std::array descriptions = {"has no operator and no vector", "has an operator but no vector",
                                   "has no operator, but has a vector",
                                   "has an operator and a vector"};

        for (std::size_t i = 0; i < dataTerms.size(); i++) {
            for (std::size_t j = 0; j < regFunc.size(); j++) {
                WHEN(std::string(
                         "Calling the conversion constructor on an OptimizationProblem with only "
                         "quadric terms. The data term ")
                     + descriptions[i] + ". The regularization term " + descriptions[j])
                {

                    RegularizationTerm reg{static_cast<data_t>(0.25), *regFunc[j]};
                    Problem prob{*dataTerms[i], reg, x0};

                    QuadricProblem converted{prob};

                    THEN("the problem can be converted and all operations yield the same result as "
                         "for the initial problem")
                    {

                        REQUIRE(converted.evaluate() == Approx(prob.evaluate()));

                        DataContainer<data_t> gradDiff =
                            prob.getGradient() - converted.getGradient();
                        REQUIRE(gradDiff.squaredL2Norm()
                                == Approx(0).margin(std::numeric_limits<data_t>::epsilon()));

                        REQUIRE(prob.getHessian().apply(x0) == converted.getHessian().apply(x0));
                    }
                }
            }
        }

        dataTerms.clear();
        dataTerms.push_back(std::move(std::make_unique<L2NormPow2<data_t>>(dd)));
        dataTerms.push_back(
            std::move(std::make_unique<L2NormPow2<data_t>>(LinearResidual<data_t>{scalingOp})));
        dataTerms.push_back(
            std::move(std::make_unique<L2NormPow2<data_t>>(LinearResidual<data_t>{dc})));
        dataTerms.push_back(
            std::move(std::make_unique<L2NormPow2<data_t>>(LinearResidual<data_t>{scalingOp, dc})));
        dataTerms.push_back(std::move(std::make_unique<WeightedL2NormPow2<data_t>>(weightingOp)));
        dataTerms.push_back(std::move(std::make_unique<WeightedL2NormPow2<data_t>>(
            LinearResidual<data_t>{scalingOp}, weightingOp)));
        dataTerms.push_back(std::move(
            std::make_unique<WeightedL2NormPow2<data_t>>(LinearResidual<data_t>{dc}, weightingOp)));
        dataTerms.push_back(std::move(std::make_unique<WeightedL2NormPow2<data_t>>(
            LinearResidual<data_t>{scalingOp, dc}, weightingOp)));

        for (const auto& dataTerm : dataTerms) {
            const auto& res = static_cast<const LinearResidual<data_t>&>(dataTerm->getResidual());
            const auto isWeighted = dynamic_cast<const WeightedL2NormPow2<data_t>*>(dataTerm.get());
            std::string probType = isWeighted ? "wls problem" : "ls problem";

            std::string desc =
                probType
                + (std::string(res.hasOperator() ? " with an operator" : " with no operator")
                   + ", ")
                + (res.hasDataVector() ? "a data vector" : "no data vector");
            WHEN("converting a " + desc + ", and no x0 to a quadric problem")
            {
                // use OptimizationProblem istead of WLSProblem as it allows for more general
                // formulations
                Problem<data_t> initialProb{*dataTerm};

                TestType prob{initialProb};

                THEN("the clone works correctly")
                {
                    auto probClone = prob.clone();

                    REQUIRE(probClone.get() != &prob);
                    REQUIRE(*probClone == prob);

                    AND_THEN("the problem behaves as expected")
                    {
                        DataContainer<data_t> zero{dd};
                        zero = 0;
                        REQUIRE(prob.getCurrentSolution() == zero);

                        REQUIRE(prob.evaluate() == 0);

                        DataContainer<data_t> grad =
                            res.hasDataVector() ? static_cast<data_t>(-1.0) * dc : zero;
                        if (res.hasDataVector() && res.hasOperator())
                            grad *= scaleFactor;
                        if (res.hasDataVector() && isWeighted)
                            grad *= weightFactor;
                        REQUIRE(prob.getGradient() == grad);

                        if (isWeighted) {
                            if (res.hasOperator()) {
                                REQUIRE(prob.getHessian()
                                        == leaf(adjoint(scalingOp) * weightingOp * scalingOp));
                            } else {
                                REQUIRE(prob.getHessian() == leaf(weightingOp));
                            }
                        } else {
                            if (res.hasOperator()) {
                                REQUIRE(prob.getHessian() == leaf(adjoint(scalingOp) * scalingOp));
                            } else {
                                REQUIRE(prob.getHessian() == leaf(Identity<data_t>{dd}));
                            }
                        }
                    }
                }
            }

            WHEN("converting a " + desc + ", and x0 to a quadric problem")
            {
                randomData.setRandom();
                DataContainer<data_t> x0{dd, randomData};

                // use OptimizationProblem istead of WLSProblem as it allows for more general
                // formulations
                Problem<data_t> initialProb{*dataTerm, x0};

                TestType prob{initialProb};

                THEN("the clone works correctly")
                {
                    auto probClone = prob.clone();

                    REQUIRE(probClone.get() != &prob);
                    REQUIRE(*probClone == prob);

                    AND_THEN("the problem behaves as expected")
                    {
                        REQUIRE(prob.getCurrentSolution() == x0);

                        if (isWeighted) {
                            if (res.hasOperator()) {
                                if (res.hasDataVector()) {
                                    REQUIRE(prob.evaluate()
                                            == Approx(0.5 * scaleFactor * scaleFactor * weightFactor
                                                          * x0.squaredL2Norm()
                                                      - scaleFactor * weightFactor * x0.dot(dc)));
                                    auto gradient = prob.getGradient();
                                    DataContainer gradientDirect =
                                        scaleFactor * (weightFactor * (scaleFactor * x0))
                                        - scaleFactor * (weightFactor * dc);
                                    for (index_t i = 0; i < gradient.getSize(); ++i)
                                        REQUIRE(gradient[i] == Approx(gradientDirect[i]));
                                } else {
                                    REQUIRE(prob.evaluate()
                                            == Approx(0.5 * scaleFactor * scaleFactor * weightFactor
                                                      * x0.squaredL2Norm()));

                                    auto gradient = prob.getGradient();
                                    DataContainer gradientDirect =
                                        scaleFactor * (weightFactor * (scaleFactor * x0));
                                    for (index_t i = 0; i < gradient.getSize(); ++i)
                                        REQUIRE(gradient[i] == Approx(gradientDirect[i]));
                                }
                                REQUIRE(prob.getHessian()
                                        == leaf(adjoint(scalingOp) * weightingOp * scalingOp));
                            } else {
                                if (res.hasDataVector()) {
                                    REQUIRE(prob.evaluate()
                                            == Approx(0.5 * weightFactor * x0.squaredL2Norm()
                                                      - weightFactor * x0.dot(dc)));
                                    REQUIRE(prob.getGradient()
                                            == weightFactor * x0 - weightFactor * dc);
                                } else {
                                    REQUIRE(prob.evaluate()
                                            == Approx(0.5 * weightFactor * x0.squaredL2Norm()));

                                    auto gradient = prob.getGradient();
                                    DataContainer gradientDirect = weightFactor * x0;
                                    for (index_t i = 0; i < gradient.getSize(); ++i)
                                        REQUIRE(gradient[i] == Approx(gradientDirect[i]));
                                }
                                REQUIRE(prob.getHessian() == leaf(weightingOp));
                            }
                        } else {
                            if (res.hasOperator()) {
                                if (res.hasDataVector()) {
                                    REQUIRE(prob.evaluate()
                                            == Approx(0.5 * scaleFactor * scaleFactor
                                                          * x0.squaredL2Norm()
                                                      - scaleFactor * x0.dot(dc)));
                                    auto gradient = prob.getGradient();
                                    DataContainer gradientDirect =
                                        scaleFactor * (scaleFactor * x0) - scaleFactor * dc;
                                    for (index_t i = 0; i < gradient.getSize(); ++i)
                                        REQUIRE(gradient[i] == Approx(gradientDirect[i]));
                                } else {
                                    REQUIRE(prob.evaluate()
                                            == Approx(0.5 * scaleFactor * scaleFactor
                                                      * x0.squaredL2Norm()));
                                    auto gradient = prob.getGradient();
                                    DataContainer gradientDirect = scaleFactor * (scaleFactor * x0);
                                    for (index_t i = 0; i < gradient.getSize(); ++i)
                                        REQUIRE(gradient[i] == Approx(gradientDirect[i]));
                                }
                                REQUIRE(prob.getHessian() == leaf(adjoint(scalingOp) * scalingOp));
                            } else {
                                if (res.hasDataVector()) {
                                    REQUIRE(prob.evaluate()
                                            == Approx(0.5 * x0.squaredL2Norm() - x0.dot(dc)));
                                    REQUIRE(prob.getGradient() == x0 - dc);
                                } else {
                                    REQUIRE(prob.evaluate() == Approx(0.5 * x0.squaredL2Norm()));
                                    REQUIRE(prob.getGradient() == x0);
                                }
                                REQUIRE(prob.getHessian() == leaf(Identity<data_t>{dd}));
                            }
                        }
                    }
                }
            }

            for (const auto& regTerm : dataTerms) {
                const auto& res =
                    static_cast<const LinearResidual<data_t>&>(regTerm->getResidual());
                const auto isWeighted =
                    dynamic_cast<const WeightedL2NormPow2<data_t>*>(regTerm.get());
                std::string regType =
                    isWeighted ? "weighted l2-norm regularizer" : "l2-norm regularizer";

                std::string desc =
                    regType
                    + (std::string(res.hasOperator() ? " with an operator" : " with no operator")
                       + ", ")
                    + (res.hasDataVector() ? "a data vector" : "no data vector");

                WHEN("converting a Tikhonov problem with a " + desc
                     + ", and no x0 to a quadric problem")
                {

                    // conversion of data terms already tested, fix to 0.5*||Ax-b||^2 for tikhonov
                    // problem tests
                    auto dataScaleFactor = static_cast<data_t>(5.0);
                    Scaling<data_t> dataScalingOp{dd, dataScaleFactor};

                    randomData.setRandom();
                    DataContainer<data_t> dataB{dd, randomData};
                    L2NormPow2<data_t> func{LinearResidual<data_t>{dataScalingOp, dataB}};

                    auto regWeight = static_cast<data_t>(0.01);
                    RegularizationTerm<data_t> reg{regWeight, *regTerm};

                    Problem<data_t> initialProb{func, reg};

                    TestType prob{initialProb};

                    THEN("the clone works correctly")
                    {
                        auto probClone = prob.clone();

                        REQUIRE(probClone.get() != &prob);
                        REQUIRE(*probClone == prob);

                        AND_THEN("the problem behaves as expected")
                        {
                            DataContainer<data_t> zero{dd};
                            zero = 0;
                            REQUIRE(prob.getCurrentSolution() == zero);

                            REQUIRE(prob.evaluate() == 0);

                            DataContainer<data_t> grad = -dataScaleFactor * dataB;

                            if (res.hasDataVector()) {
                                DataContainer<data_t> regGrad = static_cast<data_t>(-1.0) * dc;
                                if (isWeighted) {
                                    regGrad *= regWeight * weightFactor;
                                }
                                if (res.hasOperator()) {
                                    regGrad *= scaleFactor;
                                }
                                if (!isWeighted) {
                                    regGrad *= regWeight;
                                }

                                REQUIRE(prob.getGradient() == grad + regGrad);
                            } else {
                                REQUIRE(prob.getGradient() == grad);
                            }

                            if (isWeighted) {
                                Scaling<data_t> lambdaW{dd, regWeight * weightFactor};

                                if (res.hasOperator()) {
                                    REQUIRE(prob.getHessian()
                                            == leaf(adjoint(dataScalingOp) * dataScalingOp
                                                    + adjoint(scalingOp) * lambdaW * scalingOp));
                                } else {
                                    REQUIRE(
                                        prob.getHessian()
                                        == leaf(adjoint(dataScalingOp) * dataScalingOp + lambdaW));
                                }
                            } else {
                                Scaling<data_t> lambdaOp{dd, regWeight};

                                if (res.hasOperator()) {
                                    REQUIRE(prob.getHessian()
                                            == leaf(adjoint(dataScalingOp) * dataScalingOp
                                                    + lambdaOp * adjoint(scalingOp) * scalingOp));
                                } else {
                                    REQUIRE(
                                        prob.getHessian()
                                        == leaf(adjoint(dataScalingOp) * dataScalingOp + lambdaOp));
                                }
                            }
                        }
                    }
                }

                WHEN("converting a Tikhonov problem with a " + desc
                     + ", and x0 to a quadric problem")
                {
                    randomData.setRandom();
                    DataContainer<data_t> x0{dd, randomData};

                    // conversion of data terms already tested, fix to 0.5*||Ax-b||^2 for tikhonov
                    // problem tests
                    auto dataScaleFactor = static_cast<data_t>(5.0);
                    Scaling<data_t> dataScalingOp{dd, dataScaleFactor};

                    randomData.setRandom();
                    DataContainer<data_t> dataB{dd, randomData};
                    L2NormPow2<data_t> func{LinearResidual<data_t>{dataScalingOp, dataB}};

                    auto regWeight = static_cast<data_t>(0.01);
                    RegularizationTerm<data_t> reg{regWeight, *regTerm};

                    Problem<data_t> initialProb{func, reg, x0};

                    TestType prob{initialProb};

                    THEN("the clone works correctly")
                    {
                        auto probClone = prob.clone();

                        REQUIRE(probClone.get() != &prob);
                        REQUIRE(*probClone == prob);

                        AND_THEN("the problem behaves as expected")
                        {
                            REQUIRE(prob.getCurrentSolution() == x0);

                            DataContainer<data_t> Ax = dataScaleFactor * (dataScaleFactor * x0);
                            DataContainer<data_t> b = dataScaleFactor * dataB;

                            if (isWeighted) {
                                Scaling<data_t> lambdaW{dd, regWeight * weightFactor};

                                if (res.hasOperator()) {
                                    Ax += scaleFactor
                                          * (regWeight * weightFactor * (scaleFactor * x0));
                                    if (res.hasDataVector()) {
                                        b += scaleFactor * (regWeight * weightFactor * dc);
                                    }
                                    REQUIRE(prob.getHessian()
                                            == leaf(adjoint(dataScalingOp) * dataScalingOp
                                                    + adjoint(scalingOp) * lambdaW * scalingOp));
                                } else {
                                    Ax += weightFactor * regWeight * x0;
                                    if (res.hasDataVector()) {
                                        b += weightFactor * regWeight * dc;
                                    }
                                    REQUIRE(
                                        prob.getHessian()
                                        == leaf(adjoint(dataScalingOp) * dataScalingOp + lambdaW));
                                }
                            } else {
                                Scaling<data_t> lambdaOp{dd, regWeight};

                                if (res.hasOperator()) {
                                    Ax += regWeight * (scaleFactor * (scaleFactor * x0));
                                    if (res.hasDataVector()) {
                                        b += regWeight * (scaleFactor * dc);
                                    }
                                    REQUIRE(prob.getHessian()
                                            == leaf(adjoint(dataScalingOp) * dataScalingOp
                                                    + lambdaOp * adjoint(scalingOp) * scalingOp));
                                } else {
                                    Ax += regWeight * x0;
                                    if (res.hasDataVector()) {
                                        b += regWeight * dc;
                                    }
                                    REQUIRE(
                                        prob.getHessian()
                                        == leaf(adjoint(dataScalingOp) * dataScalingOp + lambdaOp));
                                }
                            }
                            REQUIRE(prob.evaluate() == Approx(0.5 * x0.dot(Ax) - x0.dot(b)));

                            auto gradient = prob.getGradient();
                            DataContainer gradientDirect = Ax - b;
                            for (index_t i = 0; i < gradient.getSize(); ++i)
                                REQUIRE(gradient[i] == Approx(gradientDirect[i]));
                        }
                    }
                }
            }
        }
    }
}