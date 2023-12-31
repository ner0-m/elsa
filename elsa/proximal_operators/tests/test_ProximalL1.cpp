#include "Error.h"
#include "ProximalL1.h"
#include "VolumeDescriptor.h"
#include "ProximalOperator.h"

#include "doctest/doctest.h"
#include <testHelpers.h>
#include <type_traits>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("proximal_operators");

TEST_CASE_TEMPLATE("ProximalL1: Testing regularity", data_t, float, double)
{
    static_assert(std::is_default_constructible_v<ProximalL1<data_t>>);
    static_assert(std::is_copy_assignable_v<ProximalL1<data_t>>);
    static_assert(std::is_copy_constructible_v<ProximalL1<data_t>>);
    static_assert(std::is_nothrow_move_assignable_v<ProximalL1<data_t>>);
    static_assert(std::is_nothrow_move_constructible_v<ProximalL1<data_t>>);
}

TEST_CASE_TEMPLATE("ProximalL1: Testing in 1D", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff({{8}});
        VolumeDescriptor volDescr(numCoeff);

        WHEN("Using ProximalL1 operator in 1D")
        {
            ProximalL1<data_t> sThrOp;

            Vector_t<data_t> data({{-2, 3, 4, -7, 7, 8, 8, 3}});
            DataContainer<data_t> x(volDescr, data);

            Vector_t<data_t> expectedRes({{0, 0, 0, -3, 3, 4, 4, 0}});
            DataContainer<data_t> expected(volDescr, expectedRes);

            THEN("Values under threshold=4 are 0 and values above are sign(v) * (abs(v) - t)")
            {
                auto res = sThrOp.apply(x, data_t{4});
                REQUIRE_UNARY(isApprox(expected, res));
            }

            THEN("Is works when accessed as ProximalOperator")
            {
                ProximalOperator<data_t> prox(sThrOp);
                auto res = prox.apply(x, data_t{4});
                REQUIRE_UNARY(isApprox(expected, res));
            }
        }
    }
}

TEST_CASE_TEMPLATE("ProximalL1: Testing in 3D", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff({{3, 2, 3}});
        VolumeDescriptor volDescr(numCoeff);

        WHEN("Using ProximalL1 operator in 3D")
        {
            ProximalL1<data_t> sThrOp;

            THEN("Values under threshold=5 are 0 and values above are sign(v) * (abs(v) - t)")
            {
                Vector_t<data_t> data({{2, 1, 6, 6, 1, 4, 2, -9, 7, 7, 7, 3, 1, 2, 8, 9, -4, 5}});
                DataContainer<data_t> dataCont(volDescr, data);

                Vector_t<data_t> expectedRes(
                    {{0, 0, 1, 1, 0, 0, 0, -4, 2, 2, 2, 0, 0, 0, 3, 4, 0, 0}});
                DataContainer<data_t> dCRes(volDescr, expectedRes);

                REQUIRE_UNARY(isApprox(dCRes, sThrOp.apply(dataCont, data_t{5})));
            }
        }
    }
}

TEST_CASE_TEMPLATE("ProximalL1: Testing general behaviour", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff({{8}});
        VolumeDescriptor volDescr(numCoeff);

        WHEN("Using ProximalL1 operator")
        {
            ProximalL1<data_t> sThrOp;
            Vector_t<data_t> zero({{0, 0, 0, 0, 0, 0, 0, 0}});
            DataContainer<data_t> dczero(volDescr, zero);

            THEN("The zero vector is returned when the zero vector is given")
            {
                Vector_t<data_t> expectedRes({{0, 0, 0, 0, 0, 0, 0, 0}});
                DataContainer<data_t> dc(volDescr, expectedRes);

                REQUIRE_UNARY(isApprox(dc, sThrOp.apply(dczero, data_t{4})));
            }

            THEN("ProximalL1 operator throws exception for differently sized v and prox")
            {
                IndexVector_t numCoeff1({{9}});
                VolumeDescriptor volDescr1(numCoeff1);
                Vector_t<data_t> data1({{0, 0, 0, 0, 0, 0, 0, 0, 0}});
                DataContainer<data_t> largerDc(volDescr1, data1);

                REQUIRE_THROWS_AS(sThrOp.apply(dczero, data_t{1}, largerDc), LogicError);
            }
        }
    }
}

TEST_SUITE_END();
