#include "ProximalL2Squared.h"
#include "VolumeDescriptor.h"

#include "doctest/doctest.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("proximal_operators");

TEST_CASE_TEMPLATE("ProximalL2: Testing regularity", data_t, float, double)
{
    static_assert(std::is_default_constructible_v<ProximalL2Squared<data_t>>);
    static_assert(std::is_copy_assignable_v<ProximalL2Squared<data_t>>);
    static_assert(std::is_copy_constructible_v<ProximalL2Squared<data_t>>);
    static_assert(std::is_nothrow_move_assignable_v<ProximalL2Squared<data_t>>);
    static_assert(std::is_nothrow_move_constructible_v<ProximalL2Squared<data_t>>);
}

TEST_CASE_TEMPLATE("ProximalL1: Testing without a given vector b", data_t, float, double)
{
    IndexVector_t numCoeff({{10}});
    VolumeDescriptor desc(numCoeff);

    Vector_t<data_t> xvec({{8.10295253, 8.93325664, 0.55758864, 9.11832862, 9.10185855, 8.03987632,
                            3.49846135, 0.55417797, 2.99925531, 5.49291213}});
    DataContainer<data_t> x(desc, xvec);

    ProximalL2Squared<data_t> prox;

    GIVEN("With a threshold parameter of 1")
    {
        auto result = prox.apply(x, 1);

        Vector_t<data_t> expectedVec(
            {{4.05147627, 4.46662832, 0.27879432, 4.55916431, 4.55092927, 4.01993816, 1.74923067,
              0.27708899, 1.49962766, 2.74645607}});
        DataContainer<data_t> expected(desc, expectedVec);

        CAPTURE(x);
        CAPTURE(expected);

        for (int i = 0; i < expected.getSize(); ++i) {
            CHECK_UNARY(checkApproxEq(expected[i], result[i]));
        }
    }

    GIVEN("With a threshold parameter of 10")
    {
        auto result = prox.apply(x, 10);

        Vector_t<data_t> expectedVec(
            {{0.73663205, 0.81211424, 0.05068988, 0.82893897, 0.82744169, 0.73089785, 0.31804194,
              0.05037982, 0.27265957, 0.49935565}});
        DataContainer<data_t> expected(desc, expectedVec);

        CAPTURE(x);
        CAPTURE(expected);

        for (int i = 0; i < expected.getSize(); ++i) {
            CHECK_UNARY(checkApproxEq(expected[i], result[i]));
        }
    }
}

TEST_CASE_TEMPLATE("ProximalL1: Testing with a given vector b", data_t, float, double)
{
    IndexVector_t numCoeff({{10}});
    VolumeDescriptor desc(numCoeff);

    Vector_t<data_t> bvec({{0.68994793, 0.76165068, 0.85495963, 0.94187384, 0.55210654, 0.41897791,
                            0.38656914, 0.65864022, 0.78059884, 0.46041161}});
    DataContainer<data_t> b(desc, bvec);

    Vector_t<data_t> xvec({{8.10295253, 8.93325664, 0.55758864, 9.11832862, 9.10185855, 8.03987632,
                            3.49846135, 0.55417797, 2.99925531, 5.49291213}});
    DataContainer<data_t> x(desc, xvec);

    ProximalL2Squared<data_t> prox(b);

    GIVEN("With a threshold parameter of 1")
    {
        auto result = prox.apply(x, 1);

        Vector_t<data_t> expectedVec({{4.39645023, 4.84745366, 0.70627414, 5.03010123, 4.82698254,
                                       4.22942712, 1.94251524, 0.6064091, 1.88992708, 2.97666187}});
        DataContainer<data_t> expected(desc, expectedVec);

        CAPTURE(b);
        CAPTURE(x);
        CAPTURE(expected);

        for (int i = 0; i < expected.getSize(); ++i) {
            CHECK_UNARY(checkApproxEq(expected[i], result[i]));
        }
    }

    GIVEN("With a threshold parameter of 10")
    {
        auto result = prox.apply(x, 10);

        Vector_t<data_t> expectedVec(
            {{1.36385744, 1.50452395, 0.82792591, 1.68518791, 1.32935672, 1.11178686, 0.66946843,
              0.64914365, 0.98229489, 0.91791166}});
        DataContainer<data_t> expected(desc, expectedVec);

        CAPTURE(b);
        CAPTURE(x);
        CAPTURE(expected);

        for (int i = 0; i < expected.getSize(); ++i) {
            CHECK_UNARY(checkApproxEq(expected[i], result[i]));
        }
    }
}

TEST_SUITE_END();
