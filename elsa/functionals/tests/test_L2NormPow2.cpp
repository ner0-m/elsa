#include <doctest/doctest.h>

#include "testHelpers.h"
#include "L2NormPow2.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("WeightedL2NormPow2: Testing without residual", data_t, float, double,
                   complex<float>, complex<double>)
{
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 13;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            L2NormPow2<data_t> func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);
            }

            THEN("a clone behaves as expected")
            {
                auto l2Clone = func.clone();

                REQUIRE_NE(l2Clone.get(), &func);
                REQUIRE_EQ(*l2Clone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<data_t> x(dd, dataVec);

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), 0.5f * dataVec.squaredNorm()));
                REQUIRE_EQ(func.getGradient(x), x);

                Identity<data_t> idOp(dd);
                REQUIRE_EQ(func.getHessian(x), leaf(idOp));
            }
        }
    }

    // GIVEN("a residual with data")
    // {
    //     IndexVector_t numCoeff(2);
    //     numCoeff << 47, 11;
    //     VolumeDescriptor dd(numCoeff);
    //
    //     Vector randomData(dd.getNumberOfCoefficients());
    //     randomData.setRandom();
    //     DataContainer<data_t> b(dd, randomData);
    //
    //     Identity<data_t> A(dd);
    //
    //     LinearResidual<data_t> linRes(A, b);
    //
    //     WHEN("instantiating")
    //     {
    //         L2NormPow2<data_t> func(linRes);
    //
    //         THEN("the functional is as expected")
    //         {
    //             REQUIRE_EQ(func.getDomainDescriptor(), dd);
    //
    //             auto* lRes = downcast_safe<LinearResidual<data_t>>(&func.getResidual());
    //             REQUIRE_UNARY(lRes);
    //             REQUIRE_EQ(*lRes, linRes);
    //         }
    //
    //         THEN("a clone behaves as expected")
    //         {
    //             auto l2Clone = func.clone();
    //
    //             REQUIRE_NE(l2Clone.get(), &func);
    //             REQUIRE_EQ(*l2Clone, func);
    //         }
    //
    //         THEN("the evaluate, gradient and Hessian work was expected")
    //         {
    //             Vector dataVec(dd.getNumberOfCoefficients());
    //             dataVec.setRandom();
    //             DataContainer<data_t> x(dd, dataVec);
    //
    //             REQUIRE_UNARY(
    //                 checkApproxEq(func.evaluate(x), 0.5f * (dataVec -
    //                 randomData).squaredNorm()));
    //
    //             DataContainer<data_t> grad(dd, (dataVec - randomData).eval());
    //             REQUIRE_EQ(func.getGradient(x), grad);
    //
    //             auto hessian = func.getHessian(x);
    //             REQUIRE_EQ(hessian.apply(x), x);
    //         }
    //     }
    // }
}

TEST_SUITE_END();
