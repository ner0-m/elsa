#include "doctest/doctest.h"

#include "Luts.hpp"
#include <algorithm>
#include <numeric>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("projectors");

TEST_CASE_TEMPLATE("Lut: Testing Lut with an array of ones", data_t, float, double)
{
    constexpr index_t size = 100;

    std::array<data_t, size> array;
    std::fill(std::begin(array), std::end(array), 1);

    Lut lut(array);

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

    std::array<data_t, size> array;
    std::iota(std::begin(array), std::end(array), 0);

    Lut lut(array);

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
            CHECK_EQ(lut(i + 0.5), i + 0.5);
        }
    }

    WHEN("Accessing with points 1/4 along the way")
    {
        for (index_t i = 0; i < size - 1; ++i) {
            CAPTURE(i);
            CAPTURE(lut(i));
            CAPTURE(lut(i + 1));
            CHECK_EQ(lut(i + 0.25), i + 0.25);
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
            CHECK_EQ(lut(i + 0.75), i + 0.75);
        }
    }
}

// Redefine GIVEN such that it's nicely usable inside an loop
#undef GIVEN
#define GIVEN(...) DOCTEST_SUBCASE((std::string("   Given: ") + std::string(__VA_ARGS__)).c_str())

TEST_CASE_TEMPLATE("ProjectedBSlineLut: testing with various degrees", data_t, float, double)
{
    for (int degree = 0; degree < 4; ++degree) {
        for (int dim = 2; dim < 4; ++dim) {
            GIVEN("BSpline Lut of degree " + std::to_string(degree)
                  + " with dim: " + std::to_string(dim))
            {
                ProjectedBSplineLut<data_t, 50> lut(dim, degree);
                ProjectedBSpline<data_t> proj_blob(dim, degree);

                for (int i = 0; i < 100; ++i) {
                    const auto distance = i / 25.;

                    CAPTURE(i);
                    CAPTURE(distance);
                    CAPTURE(lut(distance));

                    CHECK_EQ(lut(distance), Approx(proj_blob(distance)));
                    CHECK_EQ(lut(-distance), Approx(proj_blob(-distance)));
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

    ProjectedBlobLut<data_t, 100> lut(a, alpha, m);

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

TEST_SUITE_END();
