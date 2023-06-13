/**
 * @file test_EmissionLogLikelihood.cpp
 *
 * @brief Tests for the EmissionLogLikelihood class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - rewrite
 */

#include <doctest/doctest.h>

#include <cmath>
#include "testHelpers.h"
#include "EmissionLogLikelihood.h"
#include "Scaling.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("EmissionLogLikelihood: Testing with only data no residual", data_t, float,
                   double)
{
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    VolumeDescriptor dd({9, 15, 19});
    Identity<data_t> id(dd);
    GIVEN("just data (no residual)")
    {

        Vector y(dd.getNumberOfCoefficients());
        y.setRandom();
        DataContainer<data_t> dcY(dd, y);

        WHEN("instantiating without r")
        {
            EmissionLogLikelihood func(id, dcY);

            THEN("the functional is as expected")
            {
                CHECK_EQ(func.getDomainDescriptor(), dd);
            }

            THEN("a clone behaves as expected")
            {
                auto emllClone = func.clone();

                CHECK_NE(emllClone.get(), &func);
                CHECK_EQ(*emllClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                for (index_t i = 0; i < dataVec.size(); ++i) // ensure non-negative numbers
                    if (dataVec[i] < 0)
                        dataVec[i] *= -1;
                DataContainer<data_t> x(dd, dataVec);

                // compute the "true" values
                data_t trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    data_t temp = dataVec[i];
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = 1 - y[i] / temp;
                    trueScale[i] = y[i] / (temp * temp);
                }

                CHECK_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<data_t> dcTrueGrad(dd, trueGrad);
                CHECK_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                DataContainer<data_t> dcTrueScale(dd, trueScale);
                CHECK_EQ(func.getHessian(x), adjoint(id) * leaf(Scaling(dd, dcTrueScale)) * id);
            }
        }

        WHEN("instantiating with r")
        {
            Vector r(dd.getNumberOfCoefficients());
            r.setRandom();
            for (index_t i = 0; i < r.size(); ++i)
                if (r[i] < 0)
                    r[i] *= -1;
            DataContainer<data_t> dcR(dd, r);

            EmissionLogLikelihood<data_t> func(id, dcY, dcR);

            THEN("a clone behaves as expected")
            {
                auto emllClone = func.clone();

                CHECK_NE(emllClone.get(), &func);
                CHECK_EQ(*emllClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                for (index_t i = 0; i < dataVec.size(); ++i) // ensure non-negative numbers
                    if (dataVec[i] < 0)
                        dataVec[i] *= -1;
                DataContainer<data_t> x(dd, dataVec);

                // compute the "true" values
                data_t trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    auto temp = dataVec[i] + r[i];
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = 1 - y[i] / temp;
                    trueScale[i] = y[i] / (temp * temp);
                }

                CHECK_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<data_t> dcTrueGrad(dd, trueGrad);
                CHECK_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                DataContainer<data_t> dcTrueScale(dd, trueScale);
                CHECK_EQ(func.getHessian(x), adjoint(id) * leaf(Scaling(dd, dcTrueScale)) * id);
            }
        }
    }
}

TEST_SUITE_END();
