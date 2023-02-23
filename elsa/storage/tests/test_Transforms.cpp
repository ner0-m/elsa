#include "doctest/doctest.h"

#include "transforms/Absolute.h"
#include "transforms/Assign.h"
#include "transforms/Cast.h"
#include "transforms/Clip.h"
#include "transforms/Div.h"
#include "transforms/Exp.h"
#include "transforms/Extrema.h"
#include "transforms/Imag.h"
#include "transforms/InplaceAdd.h"
#include "transforms/InplaceSub.h"
#include "transforms/InplaceMul.h"
#include "transforms/InplaceDiv.h"
#include "transforms/Log.h"
#include "transforms/Real.h"
#include "transforms/Sqrt.h"
#include "transforms/Square.h"
#include "transforms/Sub.h"

#include "functions/Abs.hpp"

#include <thrust/equal.h>
#include <thrust/iterator/transform_iterator.h>

namespace doctest
{
    template <typename T>
    struct StringMaker<elsa::ContiguousStorage<T>> {
        static String convert(const elsa::ContiguousStorage<T>& vec)
        {
            std::ostringstream oss;
            oss << "[ ";
            std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(oss, " "));
            oss << "]";
            return oss.str().c_str();
        }
    };
} // namespace doctest

TEST_SUITE_BEGIN("reductions");

TEST_CASE_TEMPLATE("Assign transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::assign(src.begin(), src.end(), dst.begin());

        // Not much to check!
        CHECK_EQ(src.size(), dst.size());
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        src[0] = 2;
        elsa::assign(src.begin(), src.end(), dst.begin());

        CHECK_EQ(doctest::Approx(2), dst[0]);

        src[0] = -2;
        elsa::assign(src.begin(), src.end(), dst.begin());

        CHECK_EQ(doctest::Approx(-2), dst[0]);
    }

    GIVEN("Some container")
    {
        constexpr size_t size = 24;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        for (size_t i = 0; i < size; ++i) {
            src[i] = i;
        }

        elsa::assign(src.begin(), src.end(), dst.begin());

        CHECK_UNARY(thrust::equal(dst.begin(), dst.end(), src.begin()));
    }
}

TEST_CASE_TEMPLATE("Absolute value transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::cwiseAbs(src.begin(), src.end(), dst.begin());

        // Not much to check!
        CHECK_EQ(src.size(), dst.size());
        CHECK_UNARY(thrust::equal(dst.begin(), dst.end(), src.begin()));
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        src[0] = 2;
        elsa::cwiseAbs(src.begin(), src.end(), dst.begin());

        CHECK_EQ(doctest::Approx(2), dst[0]);

        src[0] = -2;
        elsa::cwiseAbs(src.begin(), src.end(), dst.begin());

        CHECK_EQ(doctest::Approx(2), dst[0]);
    }

    GIVEN("Some container")
    {
        constexpr ssize_t size = 24;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        for (ssize_t i = 0; i < size; ++i) {
            src[i] = i;
        }

        // store dst = abs(val)
        elsa::cwiseAbs(src.begin(), src.end(), dst.begin());

        CHECK_UNARY(thrust::equal(dst.begin(), dst.end(), src.begin()));

        for (ssize_t i = 0; i < size; ++i) {
            src[i] = -i;
        }

        auto first = thrust::make_transform_iterator(src.begin(), elsa::abs);
        // check src == abs(dst)
        CHECK_UNARY(thrust::equal(dst.begin(), dst.end(), first));
    }
}

TEST_CASE_TEMPLATE("Clip transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> v(size);

        elsa::clip(v.begin(), v.end(), v.begin(), 0, 10);

        // Not much to check!
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> v(size);

        v[0] = 2;
        elsa::clip(v.begin(), v.end(), v.begin(), 0, 10);

        CHECK_EQ(doctest::Approx(2), v[0]);

        v[0] = -2;
        elsa::clip(v.begin(), v.end(), v.begin(), 0, 10);

        CHECK_EQ(doctest::Approx(0), v[0]);

        v[0] = 20;
        elsa::clip(v.begin(), v.end(), v.begin(), 0, 10);

        CHECK_EQ(doctest::Approx(10), v[0]);
    }
}

TEST_CASE_TEMPLATE("Cast transformation", T, float, double)
{
    GIVEN("A zero sized container of type float")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<float> src(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::cast(src.begin(), src.end(), dst.begin());

        // Not much to check!
        auto first = thrust::make_transform_iterator(
            src.begin(), [] __host__ __device__(const float& val) { return static_cast<T>(val); });

        CHECK_UNARY(thrust::equal(dst.begin(), dst.end(), first));
    }

    GIVEN("A zero sized container of type float")
    {
        constexpr size_t size = 24;
        elsa::ContiguousStorage<float> src(size);
        elsa::ContiguousStorage<T> dst(size);

        for (size_t i = 0; i < size; ++i) {
            src[i] = i;
        }

        elsa::cast(src.begin(), src.end(), dst.begin());

        // Not much to check!
        auto first = thrust::make_transform_iterator(
            src.begin(), [] __host__ __device__(const float& val) { return static_cast<T>(val); });

        CHECK_UNARY(thrust::equal(dst.begin(), dst.end(), first));
    }
}

TEST_CASE_TEMPLATE("Inplace Add transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::inplaceAdd(dst.begin(), dst.end(), src.begin());

        // Not much to check!
        CHECK_EQ(src.size(), dst.size());
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        src[0] = 2;
        dst[0] = 2;
        elsa::inplaceAdd(dst.begin(), dst.end(), src.begin());

        CHECK_EQ(doctest::Approx(4), dst[0]);

        src[0] = -2;
        elsa::inplaceAdd(dst.begin(), dst.end(), src.begin());

        CHECK_EQ(doctest::Approx(2), dst[0]);
    }

    GIVEN("An arbitrarily sized container")
    {
        constexpr size_t size = 7;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        size_t i = 0;
        auto list = std::vector<T>({3.2, -4, 0, -6., 1.76, 8, 0});
        for (T elem : list) {
            src[i] = elem;
            ++i;
        }

        i = 0;
        for (T elem : list) {
            dst[i] = elem;
            ++i;
        }

        elsa::inplaceAdd(dst.begin(), dst.end(), src.begin());

        i = 0;
        for (auto elem : dst) {
            CHECK_EQ(doctest::Approx(elem), src[i] + src[i]);
            ++i;
        }
    }
}

TEST_CASE_TEMPLATE("Inplace Sub transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::inplaceSub(dst.begin(), dst.end(), src.begin());

        // Not much to check!
        CHECK_EQ(src.size(), dst.size());
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        src[0] = 2;
        dst[0] = 2;
        elsa::inplaceSub(dst.begin(), dst.end(), src.begin());

        CHECK_EQ(doctest::Approx(0), dst[0]);

        src[0] = -2;
        elsa::inplaceSub(dst.begin(), dst.end(), src.begin());

        CHECK_EQ(doctest::Approx(2), dst[0]);
    }

    GIVEN("An arbitrarily sized container")
    {
        constexpr size_t size = 7;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        size_t i = 0;
        auto list = std::vector<T>({3.2, -4, 0, -6., 1.76, 8, 0});
        for (T elem : list) {
            src[i] = elem;
            ++i;
        }

        i = 0;
        for (T elem : list) {
            dst[i] = elem;
            ++i;
        }

        elsa::inplaceSub(dst.begin(), dst.end(), src.begin());

        i = 0;
        for (auto elem : dst) {
            INFO("Pos: ", i);
            CHECK_EQ(doctest::Approx(elem), 0);
            ++i;
        }
    }
}

TEST_CASE_TEMPLATE("Inplace Mul transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::inplaceMul(dst.begin(), dst.end(), src.begin());

        // Not much to check!
        CHECK_EQ(src.size(), dst.size());
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        src[0] = 2;
        dst[0] = 2;
        elsa::inplaceMul(dst.begin(), dst.end(), src.begin());

        CHECK_EQ(doctest::Approx(4), dst[0]);

        src[0] = -2;
        elsa::inplaceMul(dst.begin(), dst.end(), src.begin());

        CHECK_EQ(doctest::Approx(-8), dst[0]);
    }

    GIVEN("An arbitrarily sized container")
    {
        constexpr size_t size = 7;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        size_t i = 0;
        auto list = std::vector<T>({3.2, -4, 0, -6., 1.76, 8, 0});
        for (T elem : list) {
            src[i] = elem;
            ++i;
        }

        thrust::fill_n(thrust::seq, dst.begin(), size, 1);

        elsa::inplaceMul(dst.begin(), dst.end(), src.begin());

        i = 0;
        for (auto elem : dst) {
            CHECK_EQ(doctest::Approx(elem), src[i]);
            ++i;
        }
    }
}

TEST_CASE_TEMPLATE("Inplace division transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::inplaceDiv(dst.begin(), dst.end(), src.begin());

        // Not much to check!
        CHECK_EQ(src.size(), dst.size());
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        src[0] = 2;
        dst[0] = 8;
        elsa::inplaceDiv(dst.begin(), dst.end(), src.begin());

        CHECK_EQ(doctest::Approx(4), dst[0]);

        src[0] = -2;
        elsa::inplaceDiv(dst.begin(), dst.end(), src.begin());

        CHECK_EQ(doctest::Approx(-2), dst[0]);
    }

    GIVEN("An arbitrarily sized container")
    {
        constexpr size_t size = 7;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        size_t i = 0;
        auto list = std::vector<T>({3.2, -4, 4.23, -6., 1.76, 8, -.29});
        for (T elem : list) {
            src[i] = elem;
            ++i;
        }

        thrust::fill_n(thrust::seq, dst.begin(), size, 1);

        elsa::inplaceDiv(dst.begin(), dst.end(), src.begin());

        i = 0;
        for (auto elem : dst) {
            CHECK_EQ(doctest::Approx(elem), 1 / src[i]);
            ++i;
        }
    }
}

TEST_CASE_TEMPLATE("Exp transformation", T, float, double)
{
    // GIVEN("An zero sized container")
    // {
    //     constexpr size_t size = 0;
    //     elsa::ContiguousStorage<T> src(size);
    //     elsa::ContiguousStorage<T> dst(size);
    //
    //     elsa::exp(src.begin(), src.end(), dst.end());
    //
    //     // Not much to check!
    //     CHECK_EQ(src.size(), dst.size());
    // }

    // GIVEN("An one sized container")
    // {
    //     constexpr size_t size = 1;
    //     elsa::ContiguousStorage<T> src(size);
    //     elsa::ContiguousStorage<T> dst(size);
    //
    //     src[0] = 2;
    //     elsa::exp(src.begin(), src.end(), dst.begin());
    //
    //     CHECK_EQ(std::exp(src[0]), dst[0]);
    //
    //     src[0] = -2;
    //     elsa::exp(src.begin(), src.end(), dst.begin());
    //
    //     CHECK_EQ(std::exp(src[0]), dst[0]);
    // }

    // GIVEN("An arbitrarily sized container")
    // {
    //     constexpr size_t size = 7;
    //     elsa::ContiguousStorage<T> src(size);
    //     elsa::ContiguousStorage<T> dst(size);
    //
    //     size_t i = 0;
    //     for (T elem : std::initializer_list<T>({3.2, -4, 0, -6., 1.76, 8, 0})) {
    //         src[i] = elem;
    //         ++i;
    //     }
    //
    //     elsa::exp(src.begin(), src.end(), dst.begin());
    //
    //     i = 0;
    //     for ([[maybe_unused]] auto elem : dst) {
    //         INFO("Pos: ", i);
    //         CHECK_EQ(std::exp(src[i]), dst[i]);
    //         ++i;
    //     }
    // }
}

TEST_CASE_TEMPLATE("Exp transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::log(src.begin(), src.end(), dst.end());

        // Not much to check!
        CHECK_EQ(src.size(), dst.size());
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        src[0] = 2;
        elsa::log(src.begin(), src.end(), dst.begin());

        CHECK_EQ(std::log(src[0]), dst[0]);

        src[0] = -2;
        elsa::log(src.begin(), src.end(), dst.begin());

        CHECK_UNARY(std::isnan(dst[0]));
    }

    // GIVEN("An arbitrarily sized container")
    // {
    //     constexpr size_t size = 7;
    //     elsa::ContiguousStorage<T> src(size);
    //     elsa::ContiguousStorage<T> dst(size);
    //
    //     size_t i = 0;
    //     for (T elem : std::initializer_list<T>({2, 4, 8, 16, 32, 64, 128})) {
    //         src[i] = elem;
    //         ++i;
    //     }
    //
    //     elsa::log(src.begin(), src.end(), dst.begin());
    //
    //     i = 0;
    //     for ([[maybe_unused]] auto elem : dst) {
    //         INFO("Pos: ", i);
    //         CHECK_EQ(std::log(src[i]), dst[i]);
    //         ++i;
    //     }
    // }
}

TEST_CASE_TEMPLATE("Sqrt transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::sqrt(src.begin(), src.end(), dst.end());

        // Not much to check!
        CHECK_EQ(src.size(), dst.size());
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        src[0] = 2;
        elsa::sqrt(src.begin(), src.end(), dst.begin());

        CHECK_EQ(std::sqrt(src[0]), dst[0]);

        src[0] = -2;
        elsa::sqrt(src.begin(), src.end(), dst.begin());

        CHECK_UNARY(std::isnan(dst[0]));
    }

    // GIVEN("An arbitrarily sized container")
    // {
    //     constexpr size_t size = 7;
    //     elsa::ContiguousStorage<T> src(size);
    //     elsa::ContiguousStorage<T> dst(size);
    //
    //     size_t i = 0;
    //     for (T elem : std::initializer_list<T>({2, 4, 8, 16, 32, 64, 128})) {
    //         src[i] = elem;
    //         ++i;
    //     }
    //
    //     elsa::sqrt(src.begin(), src.end(), dst.begin());
    //
    //     i = 0;
    //     for ([[maybe_unused]] auto elem : dst) {
    //         INFO("Pos: ", i);
    //         CHECK_EQ(std::sqrt(src[i]), dst[i]);
    //         ++i;
    //     }
    // }
}

TEST_CASE_TEMPLATE("Square transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::square(src.begin(), src.end(), dst.end());

        // Not much to check!
        CHECK_EQ(src.size(), dst.size());
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        src[0] = 2;
        elsa::square(src.begin(), src.end(), dst.begin());

        CHECK_EQ(doctest::Approx(src[0] * src[0]), dst[0]);

        src[0] = -2;
        elsa::square(src.begin(), src.end(), dst.begin());

        CHECK_EQ(doctest::Approx(src[0] * src[0]), dst[0]);
    }

    // GIVEN("An arbitrarily sized container")
    // {
    //     constexpr size_t size = 7;
    //     elsa::ContiguousStorage<T> src(size);
    //     elsa::ContiguousStorage<T> dst(size);
    //
    //     size_t i = 0;
    //     for (T elem : std::initializer_list<T>({2, 4, 8, 16, 32, 64, 128})) {
    //         src[i] = elem;
    //         ++i;
    //     }
    //
    //     elsa::square(src.begin(), src.end(), dst.begin());
    //
    //     i = 0;
    //     for ([[maybe_unused]] auto elem : dst) {
    //         INFO("Pos: ", i);
    //         CHECK_EQ(doctest::Approx(src[i] * src[i]), dst[i]);
    //         ++i;
    //     }
    // }
}

TEST_CASE_TEMPLATE("Real transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::square(src.begin(), src.end(), dst.end());

        // Not much to check!
        CHECK_EQ(src.size(), dst.size());
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<T> dst(size);

        src[0] = 2;
        elsa::real(src.begin(), src.end(), dst.begin());

        CHECK_EQ(doctest::Approx(src[0]), dst[0]);

        src[0] = -2;
        elsa::real(src.begin(), src.end(), dst.begin());

        CHECK_EQ(doctest::Approx(src[0]), dst[0]);
    }

    // GIVEN("An arbitrarily sized container")
    // {
    //     constexpr size_t size = 7;
    //     elsa::ContiguousStorage<T> src(size);
    //     elsa::ContiguousStorage<T> dst(size);
    //
    //     size_t i = 0;
    //     for (T elem : std::initializer_list<T>({2, 4, 8, 16, 32, 64, 128})) {
    //         src[i] = elem;
    //         ++i;
    //     }
    //
    //     elsa::real(src.begin(), src.end(), dst.begin());
    //
    //     i = 0;
    //     for ([[maybe_unused]] auto elem : dst) {
    //         INFO("Pos: ", i);
    //         CHECK_EQ(doctest::Approx(src[i]), dst[i]);
    //         ++i;
    //     }
    // }
}

TEST_CASE_TEMPLATE("Real transformation", T, thrust::complex<float>, thrust::complex<double>)
{
    using inner_t = elsa::value_type_of_t<T>;

    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<inner_t> dst(size);

        elsa::real(src.begin(), src.end(), dst.end());

        // Not much to check!
        CHECK_EQ(src.size(), dst.size());
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src(size);
        elsa::ContiguousStorage<inner_t> dst(size);

        src[0] = T(2, 2);
        elsa::real(src.begin(), src.end(), dst.begin());

        CHECK_EQ(doctest::Approx(src[0].real()), dst[0]);

        src[0] = -2;
        elsa::real(src.begin(), src.end(), dst.begin());

        CHECK_EQ(doctest::Approx(src[0].real()), dst[0]);
    }

    // GIVEN("An arbitrarily sized container")
    // {
    //     constexpr size_t size = 7;
    //     elsa::ContiguousStorage<T> src(size);
    //     elsa::ContiguousStorage<inner_t> dst(size);
    //
    //     size_t i = 0;
    //     for (T elem : std::initializer_list<T>({2, 4, 8, 16, 32, 64, 128})) {
    //         src[i] = elem;
    //         ++i;
    //     }
    //
    //     elsa::real(src.begin(), src.end(), dst.begin());
    //
    //     i = 0;
    //     for ([[maybe_unused]] auto elem : dst) {
    //         INFO("Pos: ", i);
    //         CHECK_EQ(doctest::Approx(src[i].real()), dst[i]);
    //         ++i;
    //     }
    // }
}

// TEST_CASE_TEMPLATE("sub transformation", T, float, double)
// {
//     GIVEN("An zero sized container")
//     {
//         constexpr size_t size = 0;
//         elsa::ContiguousStorage<T> src1(size);
//         elsa::ContiguousStorage<T> src2(size);
//         elsa::ContiguousStorage<T> dst(size);
//
//         elsa::sub(src1.begin(), src1.end(), src2.begin(), dst.begin());
//
//         // Not much to check!
//         CHECK_EQ(src1.size(), dst.size());
//         CHECK_EQ(src2.size(), dst.size());
//     }
//
//     GIVEN("An one sized container")
//     {
//         constexpr size_t size = 1;
//         elsa::ContiguousStorage<T> src1(size);
//         elsa::ContiguousStorage<T> src2(size);
//         elsa::ContiguousStorage<T> dst(size);
//
//         src1[0] = 4;
//         src2[0] = 2;
//         elsa::sub(src1.begin(), src1.end(), src2.begin(), dst.begin());
//
//         CHECK_EQ(doctest::Approx(2), dst[0]);
//
//         src1[0] = 4;
//         src2[0] = -2;
//         elsa::sub(src1.begin(), src1.end(), src2.begin(), dst.begin());
//
//         CHECK_EQ(doctest::Approx(-2), dst[0]);
//     }
//
//     GIVEN("An arbitrarily sized container")
//     {
//         constexpr size_t size = 7;
//         elsa::ContiguousStorage<T> src1(size);
//         elsa::ContiguousStorage<T> src2(size);
//         elsa::ContiguousStorage<T> dst(size);
//
//         size_t i = 0;
//         for (T elem : std::initializer_list<T>({3.2, -4, 0, -6., 1.76, 8, 0})) {
//             src1[i] = elem;
//             ++i;
//         }
//
//         i = 0;
//         for (T elem : std::initializer_list<T>({3.2, -4, 0, -6., 1.76, 8, 0})) {
//             src2[i] = elem;
//             ++i;
//         }
//
//         elsa::sub(src1.begin(), src1.end(), src2.begin(), dst.begin());
//
//         i = 0;
//         for (auto elem : dst) {
//             CHECK_EQ(doctest::Approx(elem), src1[i] - src2[i]);
//             ++i;
//         }
//     }
// }

TEST_CASE_TEMPLATE("div transformation", T, float, double)
{
    GIVEN("An zero sized container")
    {
        constexpr size_t size = 0;
        elsa::ContiguousStorage<T> src1(size);
        elsa::ContiguousStorage<T> src2(size);
        elsa::ContiguousStorage<T> dst(size);

        elsa::div(src1.begin(), src1.end(), src2.begin(), dst.begin());

        // Not much to check!
        CHECK_EQ(src1.size(), dst.size());
        CHECK_EQ(src2.size(), dst.size());
    }

    GIVEN("An one sized container")
    {
        constexpr size_t size = 1;
        elsa::ContiguousStorage<T> src1(size);
        elsa::ContiguousStorage<T> src2(size);
        elsa::ContiguousStorage<T> dst(size);

        src1[0] = 4;

        WHEN("Given one set of configuration")
        {
            src2[0] = 2;

            elsa::div(src1.begin(), src1.end(), src2.begin(), dst.begin());
            CHECK_EQ(doctest::Approx(2), dst[0]);
        }

        WHEN("Given one set of configuration")
        {
            src2[0] = -2;

            elsa::div(src1.begin(), src1.end(), src2.begin(), dst.begin());
            CHECK_EQ(doctest::Approx(-2), dst[0]);
        }

        // elsa::div(src1.begin(), src1.end(), 4, dst.begin());
        // CHECK_EQ(doctest::Approx(1), dst[0]);
        //
        // elsa::div(8, src1.begin(), src1.end(), dst.begin());
        // CHECK_EQ(doctest::Approx(0.5), dst[0]);
    }

    // GIVEN("An arbitrarily sized container")
    // {
    //     constexpr size_t size = 7;
    //     elsa::ContiguousStorage<T> src1(size);
    //     elsa::ContiguousStorage<T> src2(size);
    //     elsa::ContiguousStorage<T> dst(size);
    //
    //     size_t i = 0;
    //     for (T elem : std::initializer_list<T>({2, 4, 8, 16, 32, 64, 128})) {
    //         src1[i] = elem;
    //         ++i;
    //     }
    //
    //     i = 0;
    //     for (T elem : std::initializer_list<T>({2, 2, 2, 4, 4, 4, 8})) {
    //         src2[i] = elem;
    //         ++i;
    //     }
    //
    //     elsa::div(src1.begin(), src1.end(), src2.begin(), dst.begin());
    //
    //     i = 0;
    //     for (auto elem : dst) {
    //         CHECK_EQ(doctest::Approx(elem), src1[i] / src2[i]);
    //         ++i;
    //     }
    // }
}

TEST_SUITE_END();
