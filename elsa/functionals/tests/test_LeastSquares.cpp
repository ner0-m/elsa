#include <doctest/doctest.h>

#include "DataContainer.h"
#include "Scaling.h"
#include "testHelpers.h"
#include "LeastSquares.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("LeastSquares: with Identity operator", data_t, float, double)
{
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    VolumeDescriptor dd({11, 13});
    const auto size = dd.getNumberOfCoefficients();

    auto id = Identity<data_t>(dd);
    auto b = DataContainer<data_t>(dd, Vector::Random(size));

    LeastSquares<data_t> ls(id, b);

    THEN("the descriptor is as expected")
    {
        CHECK_EQ(ls.getDomainDescriptor(), dd);
    }

    THEN("Clone works as expected")
    {
        auto clone = ls.clone();

        CHECK_NE(clone.get(), &ls);
        CHECK_EQ(*clone, ls);
    }

    THEN("it evaluates to 0.5 * ||x - b||_2^2")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(b.getSize()));

        auto val = ls.evaluate(x);

        CHECK_EQ(val, doctest::Approx(0.5 * (x - b).squaredL2Norm()));
    }

    THEN("the gradient is A^T(A(x) - b)")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(b.getSize()));

        auto grad = ls.getGradient(x);
        auto expected = x - b;

        CHECK_UNARY(isApprox(grad, expected));
    }

    THEN("Hessian is A^T * A")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(b.getSize()));

        auto hessian = ls.getHessian(x);
        auto expected = leaf(adjoint(id) * id);

        CHECK_EQ(hessian, expected);
    }
}

TEST_CASE_TEMPLATE("LeastSquares: with Scaling operator", data_t, float, double)
{
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    VolumeDescriptor dd({11, 13});
    const auto size = dd.getNumberOfCoefficients();

    auto scale = Scaling<data_t>(dd, 10);
    auto b = DataContainer<data_t>(dd, Vector::Random(size));

    LeastSquares<data_t> ls(scale, b);

    THEN("the descriptor is as expected")
    {
        CHECK_EQ(ls.getDomainDescriptor(), dd);
    }

    THEN("Clone works as expected")
    {
        auto clone = ls.clone();

        CHECK_NE(clone.get(), &ls);
        CHECK_EQ(*clone, ls);
    }

    THEN("it evaluates to 0.5 * ||x - b||_2^2")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(b.getSize()));

        auto val = ls.evaluate(x);

        CHECK_EQ(val, doctest::Approx(0.5 * (10 * x - b).squaredL2Norm()));
    }

    THEN("the gradient is A^T(A(x) - b)")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(b.getSize()));

        auto grad = ls.getGradient(x);
        auto expected = 10 * ((10 * x) - b);

        CAPTURE(grad);
        CAPTURE(expected);

        CHECK_UNARY(isCwiseApprox(grad, expected));
    }

    THEN("Hessian is A^T * A")
    {
        auto x = DataContainer<data_t>(dd, Vector::Random(b.getSize()));

        auto hessian = ls.getHessian(x);
        auto expected = leaf(adjoint(scale) * scale);

        CHECK_EQ(hessian, expected);
    }
}

TEST_SUITE_END();
