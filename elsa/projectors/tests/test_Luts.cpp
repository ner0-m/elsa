#include "doctest/doctest.h"

#include "BSplines.h"
#include "Blobs.h"
#include <algorithm>
#include <numeric>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("projectors");

TEST_CASE_TEMPLATE("Lut: Testing Lut with an array of ones", data_t, float, double)
{
    constexpr index_t size = 100;

    std::array<data_t, size + 1> array;
    std::fill(std::begin(array), std::end(array), 1);

    Lut lut(std::move(array), (data_t) size);

    WHEN("Accessing with at integer values")
    {
        for (index_t i = 0; i < size; ++i) {
            CHECK_EQ(lut(i), 1);
        }
    }

    WHEN("Accessing with at midpoint values")
    {
        for (index_t i = 0; i < size - 1; ++i) {
            CAPTURE(i + 0.5);
            CHECK_EQ(lut(i + 0.5), 1);
        }
    }
}

TEST_CASE_TEMPLATE("Lut: Testing Lut with integer sequence", data_t, float, double)
{
    constexpr index_t size = 100;

    std::array<data_t, size + 1> array;
    std::iota(std::begin(array), std::end(array), 0);

    Lut lut(std::move(array), (data_t) size);

    WHEN("Accessing with at integer values")
    {
        for (index_t i = 0; i < size; ++i) {
            CHECK_EQ(lut(i), i);
        }
    }

    WHEN("Accessing with at midpoint values")
    {
        for (index_t i = 0; i < size - 1; ++i) {
            CAPTURE(i);
            CAPTURE(lut(i));
            CAPTURE(lut(i + 1));
            CHECK_EQ(Approx(lut(i + 0.5)), i + 0.5);
        }
    }

    WHEN("Accessing with points 1/4 along the way")
    {
        for (index_t i = 0; i < size - 1; ++i) {
            CAPTURE(i);
            CAPTURE(lut(i));
            CAPTURE(lut(i + 1));
            CHECK_EQ(Approx(lut(i + 0.25)), i + 0.25);
        }
    }

    WHEN("Accessing with points 1/3 along the way")
    {
        for (index_t i = 0; i < size - 1; ++i) {
            CAPTURE(i);
            CAPTURE(lut(i));
            CAPTURE(lut(i + 1));
            CHECK_EQ(Approx(lut(i + 0.33)), i + 0.33);
        }
    }

    WHEN("Accessing with points 3/4 along the way")
    {
        for (index_t i = 0; i < size - 1; ++i) {
            CAPTURE(i);
            CAPTURE(lut(i));
            CAPTURE(lut(i + 1));
            CHECK_EQ(Approx(lut(i + 0.75)), i + 0.75);
        }
    }
}

// Redefine GIVEN such that it's nicely usable inside an loop
#undef GIVEN
#define GIVEN(...) DOCTEST_SUBCASE((std::string("   Given: ") + std::string(__VA_ARGS__)).c_str())

TEST_CASE_TEMPLATE("ProjectedBSplineLut: testing with various degrees", data_t, float, double)
{
    for (int degree = 0; degree < 6; ++degree) {
        for (int dim = 2; dim < 6; ++dim) {
            GIVEN("BSpline Lut of degree " + std::to_string(degree)
                  + " with dim: " + std::to_string(dim))
            {
                ProjectedBSpline<data_t, 50> proj_spline(dim, degree);

                auto& lut = proj_spline.get_lut();
                CAPTURE(proj_spline.radius());

                for (int i = 0; i < 100; ++i) {
                    const auto distance = i / 25.;

                    CAPTURE(i);
                    CAPTURE(distance);
                    CAPTURE(lut(distance));

                    CHECK_EQ(lut(distance), Approx(proj_spline(distance)).epsilon(1e-3));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("ProjectedBlobLut: ", data_t, float, double)
{
    const auto a = 2;
    const auto alpha = 10.83;
    const auto m = 2;

    std::array<double, 101> expected{1.3671064952680276,
                                     1.366209140520679,
                                     1.3635202864368146,
                                     1.3590495461221548,
                                     1.3528128836429958,
                                     1.344832527778225,
                                     1.3351368521026497,
                                     1.323760222223995,
                                     1.3107428112196733,
                                     1.296130384534639,
                                     1.2799740558068384,
                                     1.2623300152802015,
                                     1.2432592326454344,
                                     1.2228271363144718,
                                     1.2011032712842662,
                                     1.178160937878318,
                                     1.1540768137691881,
                                     1.12893056178116,
                                     1.1028044260488254,
                                     1.0757828191638883,
                                     1.0479519029788988,
                                     1.0193991657525119,
                                     0.9902129983165086,
                                     0.9604822719203052,
                                     0.930295920364307,
                                     0.8997425289700537,
                                     0.868909932853028,
                                     0.8378848268644085,
                                     0.80675238945173,
                                     0.7755959225566695,
                                     0.7444965095220905,
                                     0.7135326928216648,
                                     0.6827801732549599,
                                     0.6523115320707827,
                                     0.6221959772930218,
                                     0.5924991153282746,
                                     0.5632827487344612,
                                     0.5346047008265601,
                                     0.5065186675909519,
                                     0.47907409717543664,
                                     0.4523160970195944,
                                     0.426285368491199,
                                     0.40101816870068707,
                                     0.37654629897855096,
                                     0.35289711932154216,
                                     0.33009358794397464,
                                     0.30815432491147815,
                                     0.28709369868739343,
                                     0.2669219342875695,
                                     0.24764524161852372,
                                     0.22926596246745282,
                                     0.2117827345210135,
                                     0.1951906707135796,
                                     0.1794815521451009,
                                     0.1646440327638897,
                                     0.15066385398062748,
                                     0.13752406736650655,
                                     0.12520526359037534,
                                     0.11368580576665363,
                                     0.1029420654170972,
                                     0.09294865929450916,
                                     0.08367868537452354,
                                     0.0751039563916854,
                                     0.0671952293773174,
                                     0.05992242974801804,
                                     0.053254868594009144,
                                     0.04716145192475515,
                                     0.04161088074412672,
                                     0.036571840947646775,
                                     0.03201318215877265,
                                     0.027904084748497336,
                                     0.024214214411544636,
                                     0.020913863801848218,
                                     0.017974080858669254,
                                     0.015366783581446854,
                                     0.013064861135202048,
                                     0.01104226128798172,
                                     0.009274064296470727,
                                     0.007736543464620423,
                                     0.006407212702138576,
                                     0.005264861504239613,
                                     0.004289577860543678,
                                     0.003462759678911652,
                                     0.0027671153788854513,
                                     0.0021866543689359847,
                                     0.0017066681716703673,
                                     0.0013137030013652602,
                                     0.000995524628597278,
                                     0.0007410763873099119,
                                     0.0005404311903714976,
                                     0.0003847384204545552,
                                     0.0002661665535932496,
                                     0.0001778423521946198,
                                     0.00011378743045824658,
                                     6.885294293780879e-05,
                                     3.865306353687595e-05,
                                     1.9497773431607567e-05,
                                     8.325160241062897e-06,
                                     2.632583006403572e-06,
                                     4.029329203627377e-07,
                                     0};

    ProjectedBlob<data_t, 101> proj_blob(a, alpha, m);
    auto& lut = proj_blob.get_lut();

    CAPTURE(a);
    CAPTURE(alpha);
    CAPTURE(m);

    for (int i = 0; i < 99; ++i) {
        const auto distance = i / 50.;

        CAPTURE(i);
        CAPTURE(distance);
        CHECK_EQ(Approx(lut(distance)).epsilon(0.000001), expected[i]);
    }
}

TEST_CASE_TEMPLATE("ProjectedBSplineLut: ", data_t, float, double)
{
    const auto order = 3;
    const auto dim = 2;
    const size_t TABLE_SIZE = 101;

    std::array<double, TABLE_SIZE> expected{
        0.6666666666666666,     0.6662706666666667,    0.6650986666666668,
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

    ProjectedBSpline<data_t, TABLE_SIZE> proj_spline(dim, order);

    auto& lut = proj_spline.get_lut();

    CAPTURE(order);
    CAPTURE(dim);

    for (int i = 0; i < TABLE_SIZE; ++i) {
        const auto distance = i / 50.;

        CAPTURE(i);
        CAPTURE(distance);
        CHECK_EQ(Approx(lut(distance)).epsilon(0.000001), expected[i]);
    }
}

TEST_SUITE_END();
