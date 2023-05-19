#include "Error.h"
#include "ProximalMixedL21Norm.h"
#include "VolumeDescriptor.h"
#include "IdenticalBlocksDescriptor.h"
#include "ProximalOperator.h"

#include "doctest/doctest.h"
#include "elsaDefines.h"
#include <testHelpers.h>
#include <type_traits>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("proximal_operators");

TEST_CASE_TEMPLATE("ProximalMixedL21Norm: Testing regularity", data_t, float, double)
{
    static_assert(std::is_default_constructible_v<ProximalMixedL21Norm<data_t>>);
    static_assert(std::is_copy_assignable_v<ProximalMixedL21Norm<data_t>>);
    static_assert(std::is_copy_constructible_v<ProximalMixedL21Norm<data_t>>);
    static_assert(std::is_nothrow_move_assignable_v<ProximalMixedL21Norm<data_t>>);
    static_assert(std::is_nothrow_move_constructible_v<ProximalMixedL21Norm<data_t>>);
}

TEST_CASE_TEMPLATE("ProximalMixedL21Norm: ", data_t, float, double)
{
    IndexVector_t numCoeff({{10}});
    VolumeDescriptor desc(numCoeff);
    IdenticalBlocksDescriptor blockdesc(1, desc);

    Vector_t<data_t> data({{9.00525257, 6.57715779, 8.17303488, 5.15455934, 9.88398348, 9.77293584,
                            0.42233363, 1.70329, 4.36181018, 8.0550166}});
    DataContainer<data_t> x(blockdesc, data);

    auto prox = ProximalMixedL21Norm<data_t>(1);

    GIVEN("Proximal of MixedL21 with sigma = 1")
    {
        WHEN("Threshold is 1")
        {
            DataContainer<data_t> expected(
                blockdesc,
                Vector_t<data_t>({{8.00525257, 5.57715779, 7.17303488, 4.15455934, 8.88398348,
                                   8.77293584, 0., 0.70329, 3.36181018, 7.0550166}}));

            auto result = prox.apply(x, 1);

            CAPTURE(x);
            CAPTURE(expected);
            CAPTURE(result);

            CHECK_UNARY(isApprox(expected, result));
        }

        WHEN("Threshold is 5")
        {
            DataContainer<data_t> expected(
                blockdesc, Vector_t<data_t>({{4.00525257, 1.57715779, 3.17303488, 0.15455934,
                                              4.88398348, 4.77293584, 0., 0., 0., 3.0550166}}));

            auto result = prox.apply(x, 5);

            CAPTURE(x);
            CAPTURE(expected);
            CAPTURE(result);

            CHECK_UNARY(isApprox(expected, result));
        }
    }
}

TEST_CASE_TEMPLATE("ProximalMixedL21Norm: Testing with two blocks", data_t, float, double)
{
    IndexVector_t numCoeff({{10}});
    VolumeDescriptor desc(numCoeff);
    IdenticalBlocksDescriptor blockdesc(2, desc);

    Vector_t<data_t> data(
        {{3.24913648, 7.40428097, 3.53027584, 8.35387298, 9.89774261, 0.62832139, 5.15964676,
          5.07599845, 3.02394342, 2.65935974, 4.21873499, 7.16535115, 5.58773054, 4.32746449,
          8.81802497, 8.42285999, 0.70222557, 2.24889965, 0.53578145, 7.40937567}});
    DataContainer<data_t> x(blockdesc, data);

    GIVEN("Proximal of MixedL21 with sigma = 1")
    {
        auto prox = ProximalMixedL21Norm<data_t>(1);

        WHEN("Threshold is 1")
        {
            DataContainer<data_t> expected(
                blockdesc,
                Vector_t<data_t>({{2.63895911, 6.68567485, 2.99615493, 7.46593725, 9.15108412,
                                   0.55393093, 4.1687816,  4.16171292, 2.03927959, 2.32154182,
                                   3.42647015, 6.46993383, 4.74232245, 3.86749695, 8.15281741,
                                   7.4256308,  0.56736927, 1.84382931, 0.36131899, 6.46816419}}));

            auto result = prox.apply(x, 1);

            CAPTURE(x);
            CAPTURE(expected);
            CAPTURE(result);

            CHECK_UNARY(isApprox(expected, result));
        }

        WHEN("Threshold is 5")
        {
            DataContainer<data_t> expected(
                blockdesc,
                Vector_t<data_t>({{0.19824965, 3.81125036, 0.85967125, 3.91419431, 6.16445016,
                                   0.25636909, 0.20532093, 0.50457082, 0.,         0.97027012,
                                   0.25741077, 3.68826457, 1.36069008, 2.02762682, 5.49198717,
                                   3.43671403, 0.02794408, 0.22354797, 0.,         2.70331828}}));

            auto result = prox.apply(x, 5);

            CAPTURE(x);
            CAPTURE(expected);
            CAPTURE(result);

            CHECK_UNARY(isApprox(expected, result));
        }
    }

    GIVEN("Proximal of MixedL21 with sigma = 0.5")
    {
        auto prox = ProximalMixedL21Norm<data_t>(0.5);

        WHEN("Threshold is 1")
        {
            DataContainer<data_t> expected(
                blockdesc,
                Vector_t<data_t>({{2.94404779, 7.04497791, 3.26321538, 7.90990512, 9.52441337,
                                   0.59112616, 4.66421418, 4.61885568, 2.5316115,  2.49045078,
                                   3.82260257, 6.81764249, 5.1650265,  4.09748072, 8.48542119,
                                   7.92424539, 0.63479742, 2.04636448, 0.44855022, 6.93876993}}));

            auto result = prox.apply(x, 1);

            CAPTURE(x);
            CAPTURE(expected);
            CAPTURE(result);

            CHECK_UNARY(isApprox(expected, result));
        }

        WHEN("Threshold is 5")
        {
            DataContainer<data_t> expected(
                blockdesc,
                Vector_t<data_t>({{1.72369306, 5.60776567, 2.19497355, 6.13403365, 8.03109639,
                                   0.44234524, 2.68248385, 2.79028463, 0.56228385, 1.81481493,
                                   2.23807288, 5.42680786, 3.47421031, 3.17754565, 7.15500607,
                                   5.92978701, 0.36508483, 1.23622381, 0.0996253,  5.05634698}}));

            auto result = prox.apply(x, 5);

            CAPTURE(x);
            CAPTURE(expected);
            CAPTURE(result);

            CHECK_UNARY(isApprox(expected, result));
        }
    }
}

TEST_CASE_TEMPLATE("ProximalMixedL21Norm: Testing with three blocks", data_t, float, double)
{
    IndexVector_t numCoeff({{5}});
    VolumeDescriptor desc(numCoeff);
    IdenticalBlocksDescriptor blockdesc(3, desc);

    Vector_t<data_t> data({{6.37768487, 8.86961697, 4.37377106, 8.74295425, 7.06557322, 9.19392703,
                            1.99330452, 0.17220244, 8.59833185, 0.92564478, 5.48438546, 8.24590015,
                            6.15935852, 2.8256758, 9.50309026}});
    DataContainer<data_t> x(blockdesc, data);

    GIVEN("Proximal of MixedL21 with sigma = 1")
    {
        auto prox = ProximalMixedL21Norm<data_t>(1);

        WHEN("Threshold is 1")
        {
            DataContainer<data_t> expected(
                blockdesc,
                Vector_t<data_t>({{5.86588167, 8.14695158, 3.79494437, 8.04818224, 6.47072994,
                                   8.45612305, 1.83089703, 0.1494131, 7.91505248, 0.8477157,
                                   5.0442687, 7.57405303, 5.34422644, 2.60112922, 8.70303494}}));

            auto result = prox.apply(x, 1);

            CAPTURE(x);
            CAPTURE(expected);
            CAPTURE(result);

            CHECK_UNARY(isApprox(expected, result));
        }

        WHEN("Threshold is 5")
        {
            DataContainer<data_t> expected(
                blockdesc,
                Vector_t<data_t>({{3.81866886, 5.25629004, 1.47963761, 5.26909422, 4.09135678,
                                   5.50490712, 1.1812671, 0.05825573, 5.181935, 0.5359994,
                                   3.28380163, 4.88666456, 2.08369812, 1.70294292, 5.50281365}}));

            auto result = prox.apply(x, 5);

            CAPTURE(x);
            CAPTURE(expected);
            CAPTURE(result);

            CHECK_UNARY(isApprox(expected, result));
        }
    }

    GIVEN("Proximal of MixedL21 with sigma = 7")
    {
        auto prox = ProximalMixedL21Norm<data_t>(7);

        WHEN("Threshold is 1")
        {
            DataContainer<data_t> expected(
                blockdesc,
                Vector_t<data_t>({{2.79506246, 3.81095927, 0.32198423, 3.87955021, 2.9016702,
                                   4.02929916, 0.85645213, 0.01267704, 3.81537627, 0.38014125,
                                   2.4035681, 3.54297032, 0.45343396, 1.25384977, 3.902703}}));

            auto result = prox.apply(x, 1);

            CAPTURE(x);
            CAPTURE(expected);
            CAPTURE(result);

            CHECK_UNARY(isApprox(expected, result));
        }

        WHEN("Threshold is 5")
        {
            DataContainer<data_t> expected(
                blockdesc,
                Vector_t<data_t>({{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}}));

            auto result = prox.apply(x, 5);

            CAPTURE(x);
            CAPTURE(expected);
            CAPTURE(result);

            CHECK_UNARY(isApprox(expected, result));
        }
    }
}
