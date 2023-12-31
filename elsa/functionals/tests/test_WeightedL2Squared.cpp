#include <doctest/doctest.h>

#include "testHelpers.h"
#include "WeightedL2Squared.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("WeightedL2Squared: Testing without residual", TestType, float, double,
                   complex<float>, complex<double>)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;
    using Scalar = GetFloatingPointType_t<TestType>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 7, 17;
        VolumeDescriptor dd(numCoeff);

        Vector scalingData(dd.getNumberOfCoefficients());
        scalingData.setRandom();
        DataContainer<TestType> scaleFactors(dd, scalingData);

        Scaling<TestType> scalingOp(dd, scaleFactors);

        WHEN("instantiating")
        {
            WeightedL2Squared<TestType> func(scaleFactors);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);
                REQUIRE_EQ(func.getWeightingOperator(), scalingOp);
            }

            THEN("a clone behaves as expected")
            {
                auto wl2Clone = func.clone();

                REQUIRE_NE(wl2Clone.get(), &func);
                REQUIRE_EQ(*wl2Clone, func);
            }

            Vector dataVec(dd.getNumberOfCoefficients());
            dataVec.setRandom();
            DataContainer<TestType> x(dd, dataVec);

            Vector Wx = scalingData.array() * dataVec.array();

            THEN("the evaluate works as expected")
            {
                // TODO: with complex numbers this for some reason doesn't work, the result is
                // always the negation of the expected
                if constexpr (std::is_floating_point_v<TestType>)
                    REQUIRE_UNARY(checkApproxEq(func.evaluate(x),
                                                static_cast<Scalar>(0.5) * Wx.dot(dataVec)));
            }

            THEN("the gradient works as expected")
            {
                DataContainer<TestType> dcWx(dd, Wx);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcWx));
            }

            THEN("the Hessian works as expected")
            {
                REQUIRE_EQ(func.getHessian(x), leaf(scalingOp));
            }
        }
    }
}

// TEST_CASE_TEMPLATE("WeightedL2Squared: Testing with residual", TestType, float, double,
//                    complex<float>, complex<double>)
// {
//     using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;
//     using Scalar = GetFloatingPointType_t<TestType>;
//
//     GIVEN("a residual with data")
//     {
//         // linear residual
//         IndexVector_t numCoeff(2);
//         numCoeff << 47, 11;
//         VolumeDescriptor dd(numCoeff);
//
//         Vector randomData(dd.getNumberOfCoefficients());
//         randomData.setRandom();
//         DataContainer<TestType> b(dd, randomData);
//
//         Identity<TestType> A(dd);
//
//         LinearResidual linRes(A, b);
//
//         // scaling operator
//         Vector scalingData(dd.getNumberOfCoefficients());
//         scalingData.setRandom();
//         DataContainer<TestType> scaleFactors(dd, scalingData);
//
//         Scaling scalingOp(dd, scaleFactors);
//
//         WHEN("instantiating")
//         {
//             WeightedL2Squared func(linRes, scalingOp);
//
//             THEN("the functional is as expected")
//             {
//                 REQUIRE_EQ(func.getDomainDescriptor(), dd);
//                 REQUIRE_EQ(func.getWeightingOperator(), scalingOp);
//
//                 auto* lRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
//                 REQUIRE_UNARY(lRes);
//                 REQUIRE_EQ(*lRes, linRes);
//             }
//
//             THEN("a clone behaves as expected")
//             {
//                 auto wl2Clone = func.clone();
//
//                 REQUIRE_NE(wl2Clone.get(), &func);
//                 REQUIRE_EQ(*wl2Clone, func);
//             }
//
//             Vector dataVec(dd.getNumberOfCoefficients());
//             dataVec.setRandom();
//             DataContainer<TestType> x(dd, dataVec);
//
//             Vector WRx = scalingData.array() * (dataVec - randomData).array();
//
//             THEN("the evaluate works was expected")
//             {
//                 // TODO: with complex numbers this for some reason doesn't work, the result is
//                 // always the negation of the expected
//                 if constexpr (std::is_floating_point_v<TestType>)
//                     REQUIRE_UNARY(
//                         checkApproxEq(func.evaluate(x),
//                                       static_cast<Scalar>(0.5) * WRx.dot(dataVec - randomData)));
//             }
//
//             THEN("the gradient works was expected")
//             {
//                 DataContainer<TestType> dcWRx(dd, WRx);
//                 REQUIRE_UNARY(isApprox(func.getGradient(x), dcWRx));
//             }
//
//             THEN("the Hessian works was expected")
//             {
//                 auto hessian = func.getHessian(x);
//                 Vector Wx = scalingData.array() * dataVec.array();
//                 DataContainer<TestType> dcWx(dd, Wx);
//                 REQUIRE_UNARY(isApprox(hessian.apply(x), dcWx));
//             }
//         }
//     }
// }

TEST_SUITE_END();
