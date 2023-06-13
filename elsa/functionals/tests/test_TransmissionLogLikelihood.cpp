#include <doctest/doctest.h>

#include <cmath>
#include "testHelpers.h"
#include "TransmissionLogLikelihood.h"
#include "Scaling.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("TransmissionLogLikelihood: Testing with only data no residual", data_t, float,
                   double)
{
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 3, 7, 13;
        VolumeDescriptor dd(numCoeff);

        Identity<data_t> id(dd);

        Vector y(dd.getNumberOfCoefficients());
        y.setRandom();
        DataContainer<data_t> dcY(dd, y);

        Vector b(dd.getNumberOfCoefficients());
        b.setRandom();
        // ensure b has positive values (due to log)
        for (index_t i = 0; i < dd.getNumberOfCoefficients(); ++i) {
            if (b[i] < 0)
                b[i] *= -1;
            if (b[i] == 0)
                b[i] += 1;
        }
        DataContainer<data_t> dcB(dd, b);

        WHEN("instantiating without r")
        {
            TransmissionLogLikelihood func(id, dcY, dcB);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);
            }

            THEN("a clone behaves as expected")
            {
                auto tmllClone = func.clone();

                REQUIRE_NE(tmllClone.get(), &func);
                REQUIRE_EQ(*tmllClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<data_t> x(dd, dataVec);

                // compute the "true" values
                data_t trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    auto temp = b[i] * std::exp(-dataVec[i]);
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = y[i] - temp;
                    trueScale[i] = temp;
                }

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<data_t> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                DataContainer<data_t> dcTrueScale(dd, trueScale);
                REQUIRE_EQ(func.getHessian(x), adjoint(id) * leaf(Scaling(dd, dcTrueScale)) * id);
            }
        }

        WHEN("instantiating with r")
        {
            Vector r(dd.getNumberOfCoefficients());
            r.setRandom();
            // ensure non-negative values
            for (index_t i = 0; i < r.size(); ++i)
                if (r[i] < 0)
                    r[i] *= -1;
            DataContainer<data_t> dcR(dd, r);

            TransmissionLogLikelihood func(id, dcY, dcB, dcR);

            THEN("a clone behaves as expected")
            {
                auto tmllClone = func.clone();

                REQUIRE_NE(tmllClone.get(), &func);
                REQUIRE_EQ(*tmllClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<data_t> x(dd, dataVec);

                // compute the true values
                data_t trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    auto temp = b[i] * std::exp(-dataVec[i]);
                    auto tempR = temp + r[i];
                    trueValue += tempR - y[i] * std::log(tempR);
                    trueGrad[i] = (y[i] * temp) / tempR - temp;
                    trueScale[i] = temp + (r[i] * y[i] * temp) / (tempR * tempR);
                }

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<data_t> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                DataContainer<data_t> dcTrueScale(dd, trueScale);
                REQUIRE_EQ(func.getHessian(x), adjoint(id) * leaf(Scaling(dd, dcTrueScale)) * id);
            }
        }
    }
}

TEST_SUITE_END();
