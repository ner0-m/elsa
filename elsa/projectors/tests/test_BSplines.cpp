#include "doctest/doctest.h"

#include "BSplines.h"

#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/ostr.h"
#include <utility>

using namespace elsa;

TEST_SUITE_BEGIN("projectors::BSplines");

Eigen::IOFormat vecfmt(10, 0, ", ", ", ", "", "", "[", "]");

// TEST_CASE_TEMPLATE("BSplines: 1d BSpline evaluation", data_t, float, double)
// TEST_CASE_TEMPLATE("BSplines: 1d BSpline evaluation", data_t, float)
// {
//     constexpr auto size = 11;
//
//     const auto low = -2;
//     const auto high = 2;
//
//     const auto linspace = Vector_t<data_t>::LinSpaced(size, low, high);
//     std::array<data_t, size> spline;
//
//     auto degree = 3;
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             RealVector_t coord({{linspace[i], linspace[j]}});
//             // fmt::print("bspline({}, {}) = {}\n", coord.format(vecfmt), degree,
//             //            bspline::nd_bspline_evaluate(coord, degree));
//         }
//     }
//
//     // fmt::print("{}\n", linspace.format(vecfmt));
//     // fmt::print("{}\n", spline);
// }

/// Quick and easy simpsons rule for numerical integration
template <typename data_t, typename Func>
data_t simpsons(Func f, data_t a, data_t b, int N = 50)
{
    const auto h = (b - a) / N;

    data_t sum_odds = 0.0;
    for (int i = 1; i < N; i += 2) {
        sum_odds += f(a + i * h);
    }

    data_t sum_evens = 0.0;
    for (int i = 2; i < N; i += 2) {
        sum_evens += f(a + i * h);
    }

    return (f(a) + f(b) + 2 * sum_evens + 4 * sum_odds) * h / 3;
}

/// Integrate from distance s from center, to a
/// see Fig 3.6 (p. 53) and Listing 3.1 (p. 58) in Three-Dimensional Digital Tomosynthesis by
/// Levakhina
template <typename data_t>
data_t bspline_line_integral(data_t s, index_t m, index_t dim)
{
    auto x = [=](auto t) {
        data_t res = bspline::bspline1d_evaluate<data_t>(std::sqrt(t * t + s * s), m);
        for (int i = 1; i < dim; ++i) {
            res *= bspline::bspline1d_evaluate<data_t>(0., m);
        }
        return res;
    };

    return 2 * simpsons<data_t>(x, 0, 3);
}

TEST_CASE_TEMPLATE("BSplines: 1d line integal", data_t, float)
{
    constexpr auto size = 21;

    const auto low = -2;
    const auto high = 2;

    const auto linspace = Vector_t<data_t>::LinSpaced(size, low, high);

    const int degree = 2;

    // const data_t distance = 1.f;
    //
    // fmt::print("1D bspline at distance {:4.2f}: {:8.5f}\n", distance,
    //            bspline::bspline1d_evaluate<data_t>(distance, degree));
    // fmt::print("2D bspline at distance {:4.2f}: {:8.5f}\n", distance,
    //            bspline::bspline1d_evaluate<data_t>(0, degree)
    //                * bspline::bspline1d_evaluate<data_t>(distance, degree));
    // fmt::print("2D line integral: {:8.5f}\n", bspline_line_integral<data_t>(distance, degree,
    // 2)); fmt::print("3D line integral: {:8.5f}\n", bspline_line_integral<data_t>(distance,
    // degree, 3)); MESSAGE("helloo");

    // BSpline<data_t> bspline_1d(1, degree);
    // BSpline<data_t> bspline_2d(2, degree);
    //
    // for (int i = 0; i < 101; ++i) {
    //     const data_t x = (i / 25.) - 2.;
    //
    //     CAPTURE(x);
    //     CAPTURE(bspline::bspline1d_evaluate(x, degree));
    //     CAPTURE(bspline::bspline1d_evaluate(0., degree));
    //
    //     CHECK_EQ(bspline::bspline1d_evaluate(x, degree),
    //              doctest::Approx(bspline_1d(Vector_t<data_t>({{x}}))));
    //
    //     CHECK_EQ(bspline::bspline1d_evaluate(x, degree) * bspline::bspline1d_evaluate(0.,
    //     degree),
    //              doctest::Approx(bspline_2d(Vector_t<data_t>({{x, 0}}))));
    //     CHECK_EQ(bspline::bspline1d_evaluate(x, degree) * bspline::bspline1d_evaluate(x, degree),
    //              doctest::Approx(bspline_2d(Vector_t<data_t>({{x, x}}))));
    // }

    ProjectedBSpline<data_t> proj_bspline_2d(2, degree);
    ProjectedBSpline<data_t> proj_bspline_3d(3, degree);

    CHECK_EQ(proj_bspline_2d.order(), degree);
    CHECK_EQ(proj_bspline_3d.order(), degree);
    CHECK_EQ(proj_bspline_2d.dim(), 2);
    CHECK_EQ(proj_bspline_3d.dim(), 3);

    for (int i = 0; i < 21; ++i) {
        const data_t x = (i / 5.) - 2.;

        CAPTURE(x);
        CAPTURE(bspline::bspline1d_evaluate(x, degree));
        CAPTURE(bspline::bspline1d_evaluate(0., degree));

        CHECK_EQ(bspline::bspline1d_evaluate(x, degree), doctest::Approx(proj_bspline_2d(x)));

        CHECK_EQ(bspline::bspline1d_evaluate(x, degree) * bspline::bspline1d_evaluate(0., degree),
                 doctest::Approx(proj_bspline_3d(x)));
    }
}
