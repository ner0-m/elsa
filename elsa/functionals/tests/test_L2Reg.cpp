#include <doctest/doctest.h>

#include "Complex.h"
#include "DataContainer.h"
#include "Scaling.h"
#include "testHelpers.h"
#include "L2Reg.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("L2Reg: Testing without a b", data_t, float, double, complex<float>,
                   complex<double>)
{
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    VolumeDescriptor dd({11, 13});
    const auto size = dd.getNumberOfCoefficients();

    L2Reg<data_t> l2reg(dd);

    THEN("the descriptor is as expected")
    {
        CHECK_EQ(l2reg.getDomainDescriptor(), dd);
    }

    THEN("Clone works as expected")
    {
        auto clone = l2reg.clone();

        CHECK_NE(clone.get(), &l2reg);
        CHECK_EQ(*clone, l2reg);
    }

    THEN("it evaluates to || x ||_2^2")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto val = l2reg.evaluate(x);

        CHECK_UNARY(checkApproxEq(val, 0.5 * x.squaredL2Norm()));
    }

    THEN("the gradient is x")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto grad = l2reg.getGradient(x);

        CHECK_UNARY(isApprox(grad, x));
    }

    THEN("Hessian is I")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto hessian = l2reg.getHessian(x);
        auto expected = leaf(Identity<data_t>(dd));

        CHECK_EQ(hessian, expected);
    }
}

TEST_CASE_TEMPLATE("L2Reg: Testing with a A", data_t, float, double, complex<float>,
                   complex<double>)
{
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    VolumeDescriptor dd({11, 13});
    const auto size = dd.getNumberOfCoefficients();

    const auto id = Identity<data_t>(dd);

    L2Reg<data_t> l2reg(id);

    THEN("the descriptor is as expected")
    {
        CHECK_EQ(l2reg.getDomainDescriptor(), dd);
    }

    THEN("Clone works as expected")
    {
        auto clone = l2reg.clone();

        CHECK_NE(clone.get(), &l2reg);
        CHECK_EQ(*clone, l2reg);
    }

    THEN("it evaluates to 0.5 * || I(x) ||_2^2")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto val = l2reg.evaluate(x);

        CHECK_UNARY(checkApproxEq(val, 0.5 * x.squaredL2Norm()));
    }

    THEN("the gradient is x")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto grad = l2reg.getGradient(x);

        CHECK_UNARY(isApprox(grad, x));
    }

    THEN("Hessian is I^T * I")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(dd.getNumberOfCoefficients()));

        auto hessian = l2reg.getHessian(x);
        auto expected = leaf(adjoint(Identity<data_t>(dd)) * Identity<data_t>(dd));

        CHECK_EQ(hessian, expected);
    }
}

TEST_SUITE_END();
