#include "doctest/doctest.h"

#include "BSplines.h"
#include "Luts.hpp"

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
    // constexpr auto size = 21;
    // const auto low = -2;
    // const auto high = 2;
    // const auto linspace = Vector_t<data_t>::LinSpaced(size, low, high);

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

constexpr index_t size = 101;

template <typename data_t>
constexpr std::array<double, size> getExpected(int order, int dim, bool derivative = false);

TEST_CASE_TEMPLATE("BSpline: Test evaluation of bspline derivative", data_t, float, double)
{
    ;
    const auto order = 3;
    const auto dim = 2;

    CAPTURE(order);
    CAPTURE(dim);

    auto expected = getExpected<data_t>(order, dim);
    auto expectedDerivative = getExpected<data_t>(order, dim, true);

    ProjectedBSpline<data_t> bspline(dim, order);
    for (int i = 0; i < size; ++i) {
        const auto x = i / 50.0;

        CAPTURE(x);

        CHECK_EQ(doctest::Approx(bspline.derivative(x)), expectedDerivative[i]);
        CHECK_EQ(doctest::Approx(bspline(x)), expected[i]);
    }
}

TEST_CASE_TEMPLATE("BSpline: Test evaluation of bspline derivative lut", data_t, float, double)
{
    ;
    const auto order = 3;
    const auto dim = 2;

    CAPTURE(order);
    CAPTURE(dim);

    auto expectedDerivative = getExpected<data_t>(order, dim, true);

    ProjectedBSplineDerivativeLut<data_t, 100> bspline_lut(dim, order);
    for (int i = 0; i < size; ++i) {
        const auto x = i / 50.0;

        CAPTURE(x);

        CHECK_EQ(doctest::Approx(bspline_lut(x)), expectedDerivative[i]);
    }
}

template <typename data_t>
constexpr std::array<double, size> getExpected(int order, int dim, bool derivative)
{
    if (derivative) {
        if (order == 3) {
            if (dim == 2) {
                return {0.0,
                        -0.039399999999999935,
                        -0.07760000000000039,
                        -0.11460000000000042,
                        -0.1503999999999998,
                        -0.1850000000000004,
                        -0.21839999999999982,
                        -0.25060000000000043,
                        -0.2816000000000004,
                        -0.3113999999999998,
                        -0.34000000000000014,
                        -0.3673999999999999,
                        -0.3936000000000004,
                        -0.4186000000000001,
                        -0.4424000000000003,
                        -0.4649999999999999,
                        -0.4864000000000003,
                        -0.5066000000000002,
                        -0.5255999999999998,
                        -0.5433999999999996,
                        -0.5599999999999997,
                        -0.5754000000000001,
                        -0.5895999999999999,
                        -0.6025999999999998,
                        -0.6143999999999994,
                        -0.625,
                        -0.6343999999999999,
                        -0.6425999999999998,
                        -0.6496000000000002,
                        -0.6554000000000002,
                        -0.6600000000000004,
                        -0.6634000000000002,
                        -0.6656,
                        -0.6665999999999992,
                        -0.6663999999999999,
                        -0.6649999999999999,
                        -0.6624,
                        -0.6586000000000004,
                        -0.6536000000000001,
                        -0.6473999999999998,
                        -0.6400000000000003,
                        -0.6313999999999993,
                        -0.6215999999999986,
                        -0.6106,
                        -0.5984000000000003,
                        -0.5850000000000002,
                        -0.5704000000000016,
                        -0.5546,
                        -0.5375999999999994,
                        -0.5193999999999996,
                        -0.5,
                        -0.48019999999999957,
                        -0.4607999999999994,
                        -0.4417999999999988,
                        -0.4232000000000014,
                        -0.4049999999999989,
                        -0.3872000000000006,
                        -0.36979999999999896,
                        -0.35279999999999845,
                        -0.336200000000001,
                        -0.3199999999999999,
                        -0.3042,
                        -0.2887999999999984,
                        -0.2737999999999996,
                        -0.25919999999999965,
                        -0.2450000000000002,
                        -0.23120000000000057,
                        -0.2177999999999986,
                        -0.20479999999999937,
                        -0.1922000000000007,
                        -0.18000000000000194,
                        -0.16819999999999982,
                        -0.15679999999999955,
                        -0.14579999999999935,
                        -0.1352000000000008,
                        -0.125,
                        -0.11520000000000077,
                        -0.1058000000000002,
                        -0.0968000000000003,
                        -0.08819999999999975,
                        -0.07999999999999918,
                        -0.07220000000000062,
                        -0.06479999999999789,
                        -0.057800000000000296,
                        -0.05119999999999986,
                        -0.045000000000000956,
                        -0.039199999999999846,
                        -0.033799999999999386,
                        -0.028799999999998604,
                        -0.024200000000001498,
                        -0.01999999999999824,
                        -0.016199999999999715,
                        -0.012800000000002698,
                        -0.009800000000001752,
                        -0.007199999999999818,
                        -0.0050000000000004485,
                        -0.003199999999997316,
                        -0.0017999999999993577,
                        -0.0007999999999995788,
                        -0.00019999999999953388,
                        0.0};
            }
        }
    } else {
        if (order == 3) {
            if (dim == 2) {
                return {0.6666666666666666,     0.6662706666666667,    0.6650986666666668,
                        0.6631746666666667,     0.6605226666666665,    0.6571666666666666,
                        0.6531306666666663,     0.6484386666666664,    0.6431146666666666,
                        0.6371826666666665,     0.6306666666666665,    0.6235906666666666,
                        0.6159786666666667,     0.6078546666666668,    0.5992426666666665,
                        0.5901666666666665,     0.5806506666666662,    0.5707186666666668,
                        0.5603946666666667,     0.5497026666666668,    0.5386666666666665,
                        0.5273106666666669,     0.5156586666666667,    0.5037346666666666,
                        0.49156266666666676,    0.4791666666666665,    0.46657066666666686,
                        0.45379866666666707,    0.440874666666667,     0.4278226666666668,
                        0.4146666666666662,     0.4014306666666667,    0.3881386666666664,
                        0.37481466666666674,    0.36148266666666695,   0.34816666666666657,
                        0.3348906666666667,     0.3216786666666665,    0.30855466666666703,
                        0.2955426666666664,     0.28266666666666673,   0.26995066666666734,
                        0.25741866666666646,    0.24509466666666646,   0.23300266666666647,
                        0.22116666666666662,    0.2096106666666664,    0.19835866666666635,
                        0.18743466666666697,    0.1768626666666664,    0.16666666666666696,
                        0.1568653333333326,     0.14745599999999934,   0.1384306666666665,
                        0.12978133333333305,    0.1215000000000001,    0.11357866666666698,
                        0.10600933333333376,    0.09878399999999997,   0.09189466666666615,
                        0.08533333333333333,    0.07909199999999988,   0.07316266666666546,
                        0.06753733333333228,    0.06220800000000013,   0.05716666666666797,
                        0.05240533333333441,    0.04791599999999959,   0.04369066666666471,
                        0.03972133333333355,    0.03600000000000025,   0.03251866666666516,
                        0.029269333333332086,   0.026244000000000378,  0.023434666666665854,
                        0.020833333333333634,   0.018431999999999088,  0.016222666666665553,
                        0.01419733333333284,    0.012348000000000331,  0.0106666666666691,
                        0.009145333333335032,   0.007775999999997785,  0.006550666666665872,
                        0.005461333333334373,   0.004499999999999699,  0.0036586666666668932,
                        0.0029293333333331173,  0.0023039999999991956, 0.0017746666666672573,
                        0.0013333333333340192,  0.0009719999999988072, 0.0006826666666672199,
                        0.00045733333333397574, 0.0002879999999997884, 0.00016666666666681484,
                        8.533333333371473e-05,  3.599999999948089e-05, 1.066666666660332e-05,
                        1.3333333348519716e-06, -5.551115123125783e-16};
            }
        }
    }
}