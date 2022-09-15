#include "doctest/doctest.h"

#include "ContiguousStorage.h"

#include "reductions/DotProduct.h"
#include "reductions/Sum.h"
#include "reductions/Extrema.h"
#include "reductions/L0.h"
#include "reductions/L1.h"
#include "reductions/L2.h"
#include "reductions/LInf.h"

#include "functions/Abs.hpp"
#include "thrust/complex.h"

#include <vector>

TEST_SUITE_BEGIN("reductions");

TYPE_TO_STRING(thrust::complex<float>);
TYPE_TO_STRING(thrust::complex<double>);

template <class Range, class... Args>
void push(Range& r, Args... args)
{
    r.reserve(sizeof...(Args));
    (r.push_back(args), ...);
}

TEST_CASE_TEMPLATE("Reductions with a zero sized container", T, float, double,
                   thrust::complex<float>, thrust::complex<double>)
{
    constexpr std::size_t size = 0;
    elsa::ContiguousStorage<T> v(size);

    THEN("Sum return 0")
    {
        CHECK_EQ(T(0), elsa::sum(v.begin(), v.end()));
    }

    THEN("the minimum element is 0")
    {
        CHECK_EQ(T(0), elsa::minElement(v.begin(), v.end()));
    }

    THEN("the maximum element is 0")
    {
        CHECK_EQ(T(0), elsa::maxElement(v.begin(), v.end()));
    }

    THEN("the L0 norm is 0")
    {
        CHECK_EQ(T(0), elsa::l0PseudoNorm(v.begin(), v.end()));
    }

    THEN("the L1 norm is 0")
    {
        CHECK_EQ(T(0), elsa::l1Norm(v.begin(), v.end()));
    }

    THEN("the L2 norm is 0")
    {
        CHECK_EQ(T(0), elsa::l2Norm(v.begin(), v.end()));
    }

    THEN("the L-infinity norm is 0")
    {
        CHECK_EQ(T(0), elsa::lInf(v.begin(), v.end()));
    }
}

TEST_CASE_TEMPLATE("Reduction with vectors with a single element", T, float, double)
{
    constexpr std::size_t size = 1;
    elsa::ContiguousStorage<T> v(size);

    WHEN("The element is of value 2")
    {

        v[0] = 2;

        THEN("Sum returns 2")
        {
            CHECK_EQ(T(2), elsa::sum(v.begin(), v.end()));
        }

        THEN("the minimum element is 2")
        {
            CHECK_EQ(T(2), elsa::minElement(v.begin(), v.end()));
        }

        THEN("the maximum element is 2")
        {
            CHECK_EQ(T(2), elsa::maxElement(v.begin(), v.end()));
        }

        THEN("the L0 norm is 1")
        {
            CHECK_EQ(1, elsa::l0PseudoNorm(v.begin(), v.end()));
        }

        THEN("the L1 norm is 2")
        {
            CHECK_EQ(T(2), elsa::l1Norm(v.begin(), v.end()));
        }

        THEN("the L2 norm is 2")
        {
            CHECK_EQ(T(2), elsa::l2Norm(v.begin(), v.end()));
        }

        THEN("the L-infinity norm is 2")
        {
            CHECK_EQ(T(2), elsa::lInf(v.begin(), v.end()));
        }

        THEN("dot(v, v) = squaredL2Norm(v)")
        {
            CHECK_EQ(elsa::dot(v.begin(), v.end(), v.begin()),
                     elsa::squaredL2Norm(v.begin(), v.end()));
        }
    }

    WHEN("The element is of value -2")
    {
        v[0] = -2;

        THEN("Sum returns -2")
        {
            CHECK_EQ(T(-2), elsa::sum(v.begin(), v.end()));
        }

        THEN("the minimum element is -2")
        {
            CHECK_EQ(T(-2), elsa::minElement(v.begin(), v.end()));
        }

        THEN("the maximum element is -2")
        {
            CHECK_EQ(T(-2), elsa::maxElement(v.begin(), v.end()));
        }

        THEN("the L0 norm is 1")
        {
            CHECK_EQ(1, elsa::l0PseudoNorm(v.begin(), v.end()));
        }

        THEN("the L1 norm is 2")
        {
            CHECK_EQ(T(2), elsa::l1Norm(v.begin(), v.end()));
        }

        THEN("the L2 norm is 2")
        {
            CHECK_EQ(T(2), elsa::l2Norm(v.begin(), v.end()));
        }

        THEN("the L-infinity norm is 2")
        {
            CHECK_EQ(T(2), elsa::lInf(v.begin(), v.end()));
        }

        THEN("dot(v, v) = squaredL2Norm(v)")
        {
            CHECK_EQ(elsa::dot(v.begin(), v.end(), v.begin()),
                     elsa::squaredL2Norm(v.begin(), v.end()));
        }
    }
}

TEST_CASE_TEMPLATE("Reduction with vectors with a single element", T, thrust::complex<float>,
                   thrust::complex<double>)
{
    constexpr std::size_t size = 1;
    elsa::ContiguousStorage<T> v(size);

    WHEN("The element is of value 2+2i")
    {
        v[0] = T(2, 2);

        THEN("Sum returns 2+2i")
        {
            CHECK_EQ(T(2, 2), elsa::sum(v.begin(), v.end()));
        }

        THEN("the minimum element is 2+2i")
        {
            CHECK_EQ(T(2, 2), elsa::minElement(v.begin(), v.end()));
        }

        THEN("the maximum element is 2+2i")
        {
            CHECK_EQ(T(2, 2), elsa::maxElement(v.begin(), v.end()));
        }

        THEN("the L0 norm is 1")
        {
            CHECK_EQ(1, elsa::l0PseudoNorm(v.begin(), v.end()));
        }

        THEN("the L1 norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::l1Norm(v.begin(), v.end()));
        }

        THEN("the L2 norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::l2Norm(v.begin(), v.end()));
        }

        THEN("the L-infinity norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::lInf(v.begin(), v.end()));
        }
    }

    WHEN("The element is of value 2-2i")
    {
        v[0] = T(2, -2);

        THEN("Sum returns 2-2i")
        {
            CHECK_EQ(T(2, -2), elsa::sum(v.begin(), v.end()));
        }

        THEN("the minimum element is 2-2i")
        {
            CHECK_EQ(T(2, -2), elsa::minElement(v.begin(), v.end()));
        }

        THEN("the maximum element is 2-2i")
        {
            CHECK_EQ(T(2, -2), elsa::maxElement(v.begin(), v.end()));
        }

        THEN("the L0 norm is 1")
        {
            CHECK_EQ(1, elsa::l0PseudoNorm(v.begin(), v.end()));
        }

        THEN("the L1 norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::l1Norm(v.begin(), v.end()));
        }

        THEN("the L2 norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::l2Norm(v.begin(), v.end()));
        }

        THEN("the L-infinity norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::lInf(v.begin(), v.end()));
        }
    }

    WHEN("The element is of value -2+2i")
    {
        v[0] = T(-2, 2);

        THEN("Sum returns -2+2i")
        {
            CHECK_EQ(T(-2, 2), elsa::sum(v.begin(), v.end()));
        }

        THEN("the minimum element is -2+2i")
        {
            CHECK_EQ(T(-2, 2), elsa::minElement(v.begin(), v.end()));
        }

        THEN("the maximum element is -2+2i")
        {
            CHECK_EQ(T(-2, 2), elsa::maxElement(v.begin(), v.end()));
        }

        THEN("the L0 norm is 1")
        {
            CHECK_EQ(1, elsa::l0PseudoNorm(v.begin(), v.end()));
        }

        THEN("the L1 norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::l1Norm(v.begin(), v.end()));
        }

        THEN("the L2 norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::l2Norm(v.begin(), v.end()));
        }

        THEN("the L-infinity norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::lInf(v.begin(), v.end()));
        }
    }

    WHEN("The element is of value -2-2i")
    {
        v[0] = T(-2, -2);

        THEN("Sum returns -2-2i")
        {
            CHECK_EQ(T(-2, -2), elsa::sum(v.begin(), v.end()));
        }

        THEN("the minimum element is -2-2i")
        {
            CHECK_EQ(T(-2, -2), elsa::minElement(v.begin(), v.end()));
        }

        THEN("the maximum element is -2-2i")
        {
            CHECK_EQ(T(-2, -2), elsa::maxElement(v.begin(), v.end()));
        }

        THEN("the L0 norm is 1")
        {
            CHECK_EQ(1, elsa::l0PseudoNorm(v.begin(), v.end()));
        }

        THEN("the L1 norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::l1Norm(v.begin(), v.end()));
        }

        THEN("the L2 norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::l2Norm(v.begin(), v.end()));
        }

        THEN("the L-infinity norm is 2.8284")
        {
            CHECK_EQ(doctest::Approx(2.8284271247), elsa::lInf(v.begin(), v.end()));
        }
    }
}

TEST_CASE_TEMPLATE("Reductions with an arbitrarily sized container", T, float, double)
{
    constexpr auto size = 7;
    elsa::ContiguousStorage<T> v;

    push(v, 3.2, -4, 0, -6., 1.76, 8, 0);

    THEN("Sum return 2.96")
    {
        CHECK_EQ(T(2.96), elsa::sum(v.begin(), v.end()));
    }

    THEN("the minimum element is -6")
    {
        CHECK_EQ(T(-6), elsa::minElement(v.begin(), v.end()));
    }

    THEN("the maximum element is 8")
    {
        CHECK_EQ(T(8), elsa::maxElement(v.begin(), v.end()));
    }

    THEN("the L0 norm is 5")
    {
        CHECK_EQ(T(5), elsa::l0PseudoNorm(v.begin(), v.end()));
    }

    THEN("the L1 norm is 22.96")
    {
        CHECK_EQ(T(22.96), elsa::l1Norm(v.begin(), v.end()));
    }

    THEN("the L2 norm is 11.372")
    {
        CHECK_EQ(doctest::Approx(11.37266899192973), elsa::l2Norm(v.begin(), v.end()));
    }

    THEN("the L-infinity norm is 8")
    {
        CHECK_EQ(T(8), elsa::lInf(v.begin(), v.end()));
    }

    THEN("dot(v, v) = squaredL2Norm(v)")
    {
        CHECK_EQ(elsa::dot(v.begin(), v.end(), v.begin()), elsa::squaredL2Norm(v.begin(), v.end()));
    }
}

TEST_CASE_TEMPLATE("Reductions with an arbitrarily sized container", T, thrust::complex<float>,
                   thrust::complex<double>)
{
    elsa::ContiguousStorage<T> v;
    push(v, T{1.2, 0}, T{0, 0}, T{0, 0}, T{-6, 3}, T{1.76, -4.2});

    THEN("Sum return -3.04-1.2j")
    {
        const auto sum = elsa::sum(v.begin(), v.end());

        CHECK_EQ(doctest::Approx(-3.04), sum.real());
        CHECK_EQ(doctest::Approx(-1.2), sum.imag());
    }

    THEN("the minimum element is -0+0j")
    {
        CHECK_EQ(T(0, 0), elsa::minElement(v.begin(), v.end()));
    }

    THEN("the maximum element is -6-3j")
    {
        CHECK_EQ(T(-6, 3), elsa::maxElement(v.begin(), v.end()));
    }

    THEN("the L0 norm is 5")
    {
        CHECK_EQ(3, elsa::l0PseudoNorm(v.begin(), v.end()));
    }

    THEN("the L1 norm is 12.462")
    {
        CHECK_EQ(doctest::Approx(12.46205944184527), elsa::l1Norm(v.begin(), v.end()));
    }

    THEN("the L2 norm is 8.196")
    {
        CHECK_EQ(doctest::Approx(8.196194238791563), elsa::l2Norm(v.begin(), v.end()));
    }

    THEN("the L-infinity norm is 6.708")
    {
        CHECK_EQ(doctest::Approx(6.708203932499369), elsa::lInf(v.begin(), v.end()));
    }
}

TEST_CASE_TEMPLATE("Dot reduction", T, float, double)
{

    GIVEN("two empty vectors")
    {
        constexpr std::size_t size = 0;
        elsa::ContiguousStorage<T> v1(size);
        elsa::ContiguousStorage<T> v2(size);

        CHECK_EQ(doctest::Approx(0), elsa::dot(v1.begin(), v1.end(), v2.begin()));
    }

    GIVEN("Two one element vectors")
    {
        constexpr std::size_t size = 1;
        elsa::ContiguousStorage<T> v1(size, 3);
        elsa::ContiguousStorage<T> v2(size, 6);

        CHECK_EQ(doctest::Approx(18), elsa::dot(v1.begin(), v1.end(), v2.begin()));
    }

    GIVEN("two arbitrarily sized vectors")
    {
        // Just some random vector generated in Numpy
        std::vector<T> tmp1({-0.43479596, -0.84077378, -0.56089745, -0.67305551, 0.56718627,
                             -0.91609436, 0.72341932});
        std::vector<T> tmp2(
            {-0.14797024, 0.59378121, 0.56891962, -0.45510974, -0.66143606, 0.32099413, 0.7044938});

        elsa::ContiguousStorage<T> v1(tmp1.begin(), tmp1.end());
        elsa::ContiguousStorage<T> v2(tmp2.begin(), tmp2.end());

        CHECK_EQ(doctest::Approx(-0.6072641910843998), elsa::dot(v1.begin(), v1.end(), v2.begin()));
        CHECK_EQ(doctest::Approx(-0.6072641910843998), elsa::dot(v2.begin(), v2.end(), v1.begin()));
        CHECK_EQ(elsa::dot(v1.begin(), v1.end(), v2.begin()),
                 elsa::dot(v2.begin(), v2.end(), v1.begin()));
    }
}

TEST_CASE_TEMPLATE("Dot reduction", T, thrust::complex<float>, thrust::complex<double>)
{
    GIVEN("two empty vectors")
    {
        constexpr std::size_t size = 0;
        elsa::ContiguousStorage<T> v1(size);
        elsa::ContiguousStorage<T> v2(size);

        CHECK_EQ(doctest::Approx(0), elsa::dot(v1.begin(), v1.end(), v2.begin()));
    }

    GIVEN("Two one element vectors")
    {
        constexpr std::size_t size = 1;
        elsa::ContiguousStorage<T> v1(size, T(-0.75334326, 0.72289872));
        elsa::ContiguousStorage<T> v2(size, T(-0.0237914, -0.00879711));

        const auto dot = elsa::dot(v1.begin(), v1.end(), v2.begin());
        CHECK_EQ(doctest::Approx(0.0115636698497808), dot.real());
        CHECK_EQ(doctest::Approx(0.02382601560516), dot.imag());
    }

    GIVEN("two arbitrarily sized vectors")
    {
        // Just some random vector generated in Numpy
        std::vector<T> tmp1({T(-0.874387332715213, 0.10686457363237234),
                             T(-0.035806423839290336, -0.5303469064488933),
                             T(-0.9874157992807262, -0.6493415228933546),
                             T(0.015410212815738067, -0.372014836531108),
                             T(-0.04143572839867815, 0.8252786212388084),
                             T(-0.40672039560778517, -0.22691021280786128),
                             T(0.015868774025001198, 0.5462006987533641)});

        std::vector<T> tmp2({T(0.08666666609980744, -0.5679529430220576),
                             T(0.004031123831946148, -0.573396964565319),
                             T(0.003531169110162047, -0.9219506655846104),
                             T(-0.16556156152975254, 0.5255603214631046),
                             T(0.45538723198482334, 0.9150166021350228),
                             T(-0.04639249039147497, 0.16785091293532584),
                             T(0.19794691139569753, 0.9600072683610474)});

        elsa::ContiguousStorage<T> v1(tmp1.begin(), tmp1.end());
        elsa::ContiguousStorage<T> v2(tmp2.begin(), tmp2.end());

        WHEN("Computing dot(v1, v2)")
        {
            const auto dot = elsa::dot(v1.begin(), v1.end(), v2.begin());
            CHECK_EQ(doctest::Approx(1.8091410625945892), dot.real());
            CHECK_EQ(doctest::Approx(0.7837520298226706), dot.imag());
        }

        WHEN("Computing dot(v2, v1)")
        {
            const auto dot = elsa::dot(v2.begin(), v2.end(), v1.begin());
            CHECK_EQ(doctest::Approx(1.8091410625945892), dot.real());
            CHECK_EQ(doctest::Approx(-0.7837520298226706), dot.imag());
        }
    }
}

TEST_SUITE_END();
