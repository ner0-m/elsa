
#include "doctest/doctest.h"
#include "MatrixOperator.h"
#include "Matrix.h"

#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("operator");

TEST_CASE_TEMPLATE("MatrixOperator: Testing construction", data_t, float, double)
{
    linalg::Matrix<data_t> mat(2, 3, {1, -1, 3, 0, -3, 1});

    VolumeDescriptor domain({3});
    VolumeDescriptor range({2});

    CHECK_EQ(mat.rows(), range.getNumberOfCoefficients());
    CHECK_EQ(mat.cols(), domain.getNumberOfCoefficients());

    MatrixOperator<data_t> op(domain, range, mat);

    THEN("Copies are equal")
    {
        auto copy = op.clone();
        CHECK_EQ(op, *copy);
    }

    THEN("apply works")
    {
        Vector_t<data_t> vec({{2, 1, 0}});
        DataContainer<data_t> x(domain, vec);

        auto y = op.apply(x);

        REQUIRE_EQ(y.getSize(), 2);
        CHECK_EQ(y[0], 1.f);
        CHECK_EQ(y[1], -3.f);
    }

    THEN("applyAdjoint works")
    {
        Vector_t<data_t> vec({{2, 1}});
        DataContainer<data_t> x(range, vec);

        auto y = op.applyAdjoint(x);

        REQUIRE_EQ(y.getSize(), 3);
        CHECK_EQ(y[0], 2.f);
        CHECK_EQ(y[1], -5.f);
        CHECK_EQ(y[2], 7.f);
    }
}

TEST_CASE_TEMPLATE("MatrixOperator: Testing construction", data_t, float, double)
{
    linalg::Matrix<data_t> mat(4, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    VolumeDescriptor domain({3});
    VolumeDescriptor range({4});

    CHECK_EQ(mat.rows(), range.getNumberOfCoefficients());
    CHECK_EQ(mat.cols(), domain.getNumberOfCoefficients());

    MatrixOperator<data_t> op(domain, range, mat);

    THEN("Copies are equal")
    {
        auto copy = op.clone();
        CHECK_EQ(op, *copy);
    }

    THEN("Apply works")
    {
        Vector_t<data_t> vec({{-2, 1, 0}});
        DataContainer<data_t> x(domain, vec);

        auto y = op.apply(x);

        REQUIRE_EQ(y.getSize(), 4);
        CHECK_EQ(y[0], 0.f);
        CHECK_EQ(y[1], -3.f);
        CHECK_EQ(y[2], -6.f);
        CHECK_EQ(y[3], -9.f);
    }

    THEN("applyAdjoint works")
    {
        Vector_t<data_t> vec({{1, 2, 7, -4}});
        DataContainer<data_t> x(range, vec);

        auto y = op.applyAdjoint(x);

        REQUIRE_EQ(y.getSize(), 3);
        CHECK_EQ(y[0], 18.f);
        CHECK_EQ(y[1], 24.f);
        CHECK_EQ(y[2], 30.f);
    }
}

TEST_SUITE_END();
