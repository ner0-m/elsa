/**
 * @file test_QuadricProblem.cpp
 *
 * @brief Tests for the QuqricProblem class
 *
 * @author Nikola Dinev
 */

#include "doctest/doctest.h"

#include "QuadricProblem.h"
#include "Identity.h"
#include "Scaling.h"
#include "L2NormPow2.h"
#include "WeightedL2NormPow2.h"
#include "Huber.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"
#include "TypeCasts.hpp"

#include <array>

using namespace elsa;
using namespace doctest;

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_SUITE_BEGIN("problems");

TEST_CASE_TEMPLATE("QuadricProblem: Construction with a Quadric functional", TestType, float,
                   double)
{
    using data_t = TestType;

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
            QuadricProblem<data_t> prob{quad};

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer<data_t> zero(dd);
                zero = 0;

                REQUIRE_UNARY(checkApproxEq(prob.evaluate(zero), 0));
                REQUIRE_UNARY(checkApproxEq(prob.getGradient(zero), as<data_t>(-1.0) * dc));

                auto hessian = prob.getHessian(zero);
                REQUIRE_EQ(*hessian, scalingOp);
            }
        }
    }
}

TEST_CASE_TEMPLATE("QuadricProblem: with a Quadric functional with spd operator", TestType, float,
                   double)
{
    using data_t = TestType;

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("a spd operator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 11, 7;
        VolumeDescriptor dd{numCoeff};

        data_t scaleFactor = 3.0;
        Scaling<data_t> scalingOp{dd, scaleFactor};

        Eigen::Matrix<data_t, -1, 1> randomData{dd.getNumberOfCoefficients()};
        randomData.setRandom();
        DataContainer<data_t> b{dd, randomData};

        WHEN("constructing a QuadricProblem without x0 from it")
        {
            QuadricProblem<data_t> prob{scalingOp, b, true};

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer<data_t> zero(dd);
                zero = 0;

                REQUIRE_UNARY(checkApproxEq(prob.evaluate(zero), 0));
                REQUIRE_UNARY(isApprox(prob.getGradient(zero), static_cast<data_t>(-1.0) * b));

                auto hessian = prob.getHessian(zero);
                REQUIRE_EQ(*hessian, scalingOp);
            }
        }
    }
}

TEST_CASE_TEMPLATE("QuadricProblem: with a Quadric functional with non-spd operator", TestType,
                   float, double)
{
    using data_t = TestType;

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

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
            QuadricProblem<data_t> prob{scalingOp, dc, false};

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer<data_t> zero(dd);
                zero = 0;

                REQUIRE_UNARY(checkApproxEq(prob.evaluate(zero), 0));
                REQUIRE_UNARY(isApprox(prob.getGradient(zero), as<data_t>(-scaleFactor) * dc));

                auto hessian = prob.getHessian(zero);
                REQUIRE_EQ(*hessian, adjoint(scalingOp) * scalingOp);
            }
        }
    }
}

TEST_CASE_TEMPLATE("QuadricProblem: with a different optimization problems", TestType, float,
                   double)
{
    using data_t = TestType;

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

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
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS(QuadricProblem{prob});
            }
        }

        WHEN("trying to convert a problem with a non-quadric non-wls regularization term")
        {
            Problem prob{L2NormPow2<data_t>{dd},
                         RegularizationTerm{static_cast<data_t>(0.5), Huber<data_t>{dd}}};
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS(QuadricProblem{prob});
            }
        }

        WHEN("converting an optimization problem that has only a quadric data term")
        {
            randomData.setRandom();
            DataContainer<data_t> x0{dd, randomData};

            Problem<data_t> initialProb{Quadric<data_t>{weightingOp, dc}};

            QuadricProblem<data_t> prob{initialProb};
            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE_UNARY(checkApproxEq(prob.evaluate(x0),
                                            as<data_t>(0.5 * weightFactor) * x0.squaredL2Norm()
                                                - x0.dot(dc)));
                REQUIRE_UNARY(isApprox(prob.getGradient(x0), (weightFactor * x0) - dc));

                auto hessian = prob.getHessian(x0);
                REQUIRE_EQ(*hessian, weightingOp);
            }
        }

        WHEN("calling the conversion constructor on a QadricProblem")
        {
            randomData.setRandom();
            DataContainer<data_t> x0{dd, randomData};

            QuadricProblem<data_t> initialProb{Quadric<data_t>{weightingOp, dc}};

            QuadricProblem<data_t> prob{initialProb};
            THEN("conversion yields the same result as cloning")
            {
                REQUIRE_EQ(*initialProb.clone(), prob);
            }

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE_UNARY(checkApproxEq(prob.evaluate(x0),
                                            as<data_t>(0.5 * weightFactor) * x0.squaredL2Norm()
                                                - x0.dot(dc)));
                REQUIRE_UNARY(isApprox(prob.getGradient(x0), (weightFactor * x0) - dc));

                auto hessian = prob.getHessian(x0);
                REQUIRE_EQ(*hessian, weightingOp);
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
                WHEN("Calling the conversion constructor with an OptimizationProblem with only "
                     "quadric terms")
                {
                    INFO("The data term: ", descriptions[i]);
                    INFO("The regularization term: ", descriptions[j]);

                    RegularizationTerm reg{static_cast<data_t>(0.25), *regFunc[j]};
                    Problem prob{*dataTerms[i], reg};

                    QuadricProblem converted{prob};

                    THEN("the problem can be converted and all operations yield the same result as "
                         "for the initial problem")
                    {

                        REQUIRE_UNARY(checkApproxEq(converted.evaluate(x0), prob.evaluate(x0)));

                        DataContainer<data_t> gradDiff =
                            prob.getGradient(x0) - converted.getGradient(x0);
                        REQUIRE_UNARY(checkApproxEq(gradDiff.squaredL2Norm(), 0));

                        REQUIRE_UNARY(isApprox(prob.getHessian(x0)->apply(x0),
                                               converted.getHessian(x0)->apply(x0)));
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
            const auto isWeighted = is<WeightedL2NormPow2<data_t>>(dataTerm.get());
            std::string probType = isWeighted ? "wls problem" : "ls problem";

            std::string desc =
                probType
                + (std::string(res.hasOperator() ? " with an operator" : " with no operator")
                   + ", ")
                + (res.hasDataVector() ? "a data vector" : "no data vector");
            WHEN("Converting a quadric problem with no x0")
            {
                INFO("Converting a ", desc, "and no x0 to a quadric problem");
                // use OptimizationProblem instead of WLSProblem as it allows for more general
                // formulations
                Problem<data_t> initialProb{*dataTerm};

                QuadricProblem<data_t> prob{initialProb};

                THEN("the clone works correctly")
                {
                    auto probClone = prob.clone();

                    REQUIRE_NE(probClone.get(), &prob);
                    REQUIRE_EQ(*probClone, prob);

                    AND_THEN("the problem behaves as expected")
                    {
                        DataContainer<data_t> zero{dd};
                        zero = 0;
                        REQUIRE_UNARY(checkApproxEq(prob.evaluate(zero), 0));

                        DataContainer<data_t> grad =
                            res.hasDataVector() ? static_cast<data_t>(-1.0) * dc : zero;
                        if (res.hasDataVector() && res.hasOperator())
                            grad *= scaleFactor;
                        if (res.hasDataVector() && isWeighted)
                            grad *= weightFactor;
                        REQUIRE_UNARY(isApprox(prob.getGradient(zero), grad));

                        if (isWeighted) {
                            if (res.hasOperator()) {
                                REQUIRE_EQ(*prob.getHessian(zero),
                                           scalingOp * weightingOp * scalingOp);
                            } else {
                                REQUIRE_EQ(*prob.getHessian(zero), weightingOp);
                            }
                        } else {
                            if (res.hasOperator()) {
                                REQUIRE_EQ(*prob.getHessian(zero), adjoint(scalingOp) * scalingOp);
                            } else {
                                REQUIRE_EQ(*prob.getHessian(zero), Identity<data_t>{dd});
                            }
                        }
                    }
                }
            }

            for (const auto& regTerm : dataTerms) {
                const auto& res =
                    static_cast<const LinearResidual<data_t>&>(regTerm->getResidual());
                const auto isWeighted = is<WeightedL2NormPow2<data_t>>(regTerm.get());
                std::string regType =
                    isWeighted ? "weighted l2-norm regularizer" : "l2-norm regularizer";

                std::string desc =
                    regType
                    + (std::string(res.hasOperator() ? " with an operator" : " with no operator")
                       + ", ")
                    + (res.hasDataVector() ? "a data vector" : "no data vector");

                WHEN("Converting a Tikhonov problem with no x0 to a quadric problem")
                {

                    INFO("Converting a Tikhonov problem with a ", desc,
                         "and no x0 to a quadric problem");

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

                    QuadricProblem<data_t> prob{initialProb};

                    THEN("the clone works correctly")
                    {
                        auto probClone = prob.clone();

                        REQUIRE_NE(probClone.get(), &prob);
                        REQUIRE_EQ(*probClone, prob);

                        AND_THEN("the problem behaves as expected")
                        {
                            DataContainer<data_t> zero{dd};
                            zero = 0;

                            REQUIRE_UNARY(checkApproxEq(prob.evaluate(zero), 0));

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

                                REQUIRE_UNARY(isApprox(prob.getGradient(zero), grad + regGrad));
                            } else {
                                REQUIRE_UNARY(isApprox(prob.getGradient(zero), grad));
                            }

                            if (isWeighted) {
                                Scaling<data_t> lambdaW{dd, regWeight * weightFactor};

                                if (res.hasOperator()) {
                                    REQUIRE_EQ(*prob.getHessian(zero),
                                               adjoint(dataScalingOp) * dataScalingOp
                                                   + adjoint(scalingOp) * lambdaW * scalingOp);
                                } else {
                                    REQUIRE_EQ(*prob.getHessian(zero),
                                               adjoint(dataScalingOp) * dataScalingOp + lambdaW);
                                }
                            } else {
                                Scaling<data_t> lambdaOp{dd, regWeight};

                                if (res.hasOperator()) {
                                    REQUIRE_EQ(*prob.getHessian(zero),
                                               adjoint(dataScalingOp) * dataScalingOp
                                                   + lambdaOp * adjoint(scalingOp) * scalingOp);
                                } else {
                                    REQUIRE_EQ(*prob.getHessian(zero),
                                               adjoint(dataScalingOp) * dataScalingOp + lambdaOp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_SUITE_END();
