#include <doctest/doctest.h>

#include "Complex.h"
#include "DataContainer.h"
#include "Scaling.h"
#include "testHelpers.h"
#include "L2Squared.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("L2Squared: Testing without a b", data_t, float, double, complex<float>,
                   complex<double>)
{
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    VolumeDescriptor dd({11, 13});
    const auto size = dd.getNumberOfCoefficients();

    L2Squared<data_t> l2sqr(dd);

    THEN("the descriptor is as expected")
    {
        CHECK_EQ(l2sqr.getDomainDescriptor(), dd);
    }

    THEN("Clone works as expected")
    {
        auto clone = l2sqr.clone();

        CHECK_NE(clone.get(), &l2sqr);
        CHECK_EQ(*clone, l2sqr);
    }

    THEN("it evaluates to || x ||_2^2")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto val = l2sqr.evaluate(x);

        CHECK_UNARY(checkApproxEq(val, 0.5 * x.squaredL2Norm()));
    }

    THEN("the gradient is x")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto grad = l2sqr.getGradient(x);

        CHECK_UNARY(isApprox(grad, x));
    }

    THEN("Hessian is I")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto hessian = l2sqr.getHessian(x);
        auto expected = leaf(Identity<data_t>(dd));

        CHECK_EQ(hessian, expected);
    }
}

TEST_CASE_TEMPLATE("L2Squared: Testing with a b", data_t, float, double, complex<float>,
                   complex<double>)
{
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    VolumeDescriptor dd({11, 13});
    const auto size = dd.getNumberOfCoefficients();

    const auto b = DataContainer<data_t>(dd, Vector::Random(size));

    L2Squared<data_t> l2sqr(b);

    THEN("the descriptor is as expected")
    {
        CHECK_EQ(l2sqr.getDomainDescriptor(), dd);
    }

    THEN("Clone works as expected")
    {
        auto clone = l2sqr.clone();

        CHECK_NE(clone.get(), &l2sqr);
        CHECK_EQ(*clone, l2sqr);
    }

    THEN("it evaluates to 0.5 * || x - b ||_2^2")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto val = l2sqr.evaluate(x);

        CHECK_UNARY(checkApproxEq(val, 0.5 * (x - b).squaredL2Norm()));
    }

    THEN("the gradient is x - b")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto grad = l2sqr.getGradient(x);

        CHECK_UNARY(isApprox(grad, x - b));
    }

    THEN("Hessian is I")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto hessian = l2sqr.getHessian(x);
        auto expected = leaf(Identity<data_t>(dd));

        CHECK_EQ(hessian, expected);
    }
}

TEST_SUITE_END();
