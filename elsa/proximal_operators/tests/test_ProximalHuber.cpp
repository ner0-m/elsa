#include "Error.h"
#include "ProximalHuber.h"
#include "VolumeDescriptor.h"
#include "ProximalOperator.h"

#include "doctest/doctest.h"
#include "elsaDefines.h"
#include <testHelpers.h>
#include <type_traits>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("proximal_operators");

TEST_CASE_TEMPLATE("ProximalHuber: Testing regularity", data_t, float, double)
{
    static_assert(std::is_default_constructible_v<ProximalHuber<data_t>>);
    static_assert(std::is_copy_assignable_v<ProximalHuber<data_t>>);
    static_assert(std::is_copy_constructible_v<ProximalHuber<data_t>>);
    static_assert(std::is_nothrow_move_assignable_v<ProximalHuber<data_t>>);
    static_assert(std::is_nothrow_move_constructible_v<ProximalHuber<data_t>>);
}

TEST_CASE_TEMPLATE("ProximalHuber: Testing with delta = 0.001", data_t, float, double)
{
    IndexVector_t numCoeff({{10}});
    VolumeDescriptor volDescr(numCoeff);

    Vector_t<data_t> data({{0.89777046, 0.39362135, 0.08189309, 0.76117048, 0.99388652, 0.98390482,
                            0.70082626, 0.23339446, 0.26414902, 0.65125004}});
    DataContainer<data_t> x(volDescr, data);

    ProximalHuber<data_t> prox(0.001);

    WHEN("Threshold is 1")
    {
        DataContainer<data_t> expected(
            volDescr,
            Vector_t<data_t>({{0.4765385, 0.20893506, 0.04346903, 0.40403094, 0.52755711, 0.5222588,
                               0.37200009, 0.12388628, 0.14021087, 0.34568493}}));

        auto result = prox.apply(x, 1);

        CAPTURE(expected);
        CAPTURE(result);

        CHECK_UNARY(isApprox(expected, result, 0.001));
    }

    WHEN("Threshold is 2")
    {
        DataContainer<data_t> expected(
            volDescr,
            Vector_t<data_t>({{0.05530654, 0.02424878, 0.00504497, 0.04689139, 0.0612277,
                               0.06061279, 0.04317393, 0.01437811, 0.01627272, 0.04011982}}));
        auto result = prox.apply(x, 2);

        CAPTURE(expected);
        CAPTURE(result);

        CHECK_UNARY(isApprox(expected, result, 0.01));
    }
}

TEST_CASE_TEMPLATE("ProximalHuber: Testing with delta = 0.1", data_t, float, double)
{
    IndexVector_t numCoeff({{10}});
    VolumeDescriptor volDescr(numCoeff);

    Vector_t<data_t> data({{8.15403792, 6.6443125, 4.61017835, 6.08210491, 1.89635934, 0.46735998,
                            7.83014697, 2.65557841, 3.17244971, 4.25099761}});
    DataContainer<data_t> x(volDescr, data);

    ProximalHuber<data_t> prox(0.1);

    WHEN("Threshold is 1")
    {
        DataContainer<data_t> expected(
            volDescr,
            Vector_t<data_t>({{7.65705689, 6.23934784, 4.32919227, 5.71140627, 1.78077801,
                               0.43887483, 7.3529068, 2.49372337, 2.97909186, 3.99190326}}));

        auto result = prox.apply(x, 1);

        CAPTURE(expected);
        CAPTURE(result);

        CHECK_UNARY(isApprox(expected, result, 0.001));
    }

    WHEN("Threshold is 5")
    {
        DataContainer<data_t> expected(
            volDescr,
            Vector_t<data_t>({{5.6691328, 4.61948918, 3.20524795, 4.22861173, 1.31845266, 0.3249342,
                               5.44394611, 1.84630324, 2.20566042, 2.95552586}}));
        auto result = prox.apply(x, 5);

        CAPTURE(expected);
        CAPTURE(result);

        CHECK_UNARY(isApprox(expected, result, 0.01));
    }
}
