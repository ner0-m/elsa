#include "doctest/doctest.h"

#include "thrust/fill.h"
#include "Matrix.h"

TEST_SUITE_BEGIN("linalg");

using namespace doctest;
using namespace elsa;
using namespace elsa::linalg;

TEST_CASE_TEMPLATE("Constructing a Matrix", T, float, double)
{
    Matrix<T> m(3, 5);

    CHECK_EQ(m.rows(), 3);
    CHECK_EQ(m.cols(), 5);

    THEN("set individual values")
    {
        m(2, 4) = 123;
        CHECK_EQ(m(2, 4), 123.f);
    }

    THEN("fill with a value")
    {
        m = 1235;
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                CHECK_EQ(m(i, j), 1235.f);
            }
        }
    }

    THEN("reshape it")
    {
        m.reshape(5, 3);

        CHECK_EQ(m.rows(), 5);
        CHECK_EQ(m.cols(), 3);
    }

    THEN("resize it")
    {
        m.resize(7, 5);

        CHECK_EQ(m.rows(), 7);
        CHECK_EQ(m.cols(), 5);
    }

    THEN("Fill first row with values")
    {
        // Zero fill first
        m = 0;

        auto row = m.row(0);
        CHECK_EQ(row.size(), m.cols());

        row = 123;

        for (int j = 0; j < m.cols(); ++j) {
            CHECK_EQ(m(0, j), 123.f);
        }

        // The rest is still 0
        for (int i = 1; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                CHECK_EQ(m(i, j), 0.f);
            }
        }

        CHECK_THROWS(m.row(m.rows()));
    }

    THEN("Fill row with vector")
    {
        // Zero fill first
        m = 0.;
        Vector<T> v({0, 1, 2, 3, 4});

        auto row = m.row(0);
        CHECK_EQ(row.size(), m.cols());

        row = v;

        for (int j = 0; j < m.cols(); ++j) {
            CHECK_EQ(m(0, j), j);
        }

        // The rest is still 0
        for (int i = 1; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                CHECK_EQ(m(i, j), 0.f);
            }
        }

        CHECK_THROWS(m.row(m.rows()));
    }

    THEN("Fill first column with values")
    {
        // Zero fill first
        m = 0;

        auto col = m.col(0);
        CHECK_EQ(col.size(), m.rows());

        col = 123;

        for (int i = 0; i < m.rows(); ++i) {
            CHECK_EQ(m(i, 0), 123.f);
        }

        // The rest is still 0
        for (int j = 1; j < m.cols(); ++j) {
            for (int i = 0; i < m.rows(); ++i) {
                CHECK_EQ(m(i, j), 0.f);
            }
        }

        CHECK_THROWS(m.col(m.cols()));
    }

    THEN("Fill column with vector")
    {
        // Zero fill first
        m = 0;
        Vector<T> v({0, 1, 2});

        auto col = m.col(0);
        CHECK_EQ(col.size(), m.rows());

        col = v;

        for (int i = 0; i < m.rows(); ++i) {
            CHECK_EQ(m(i, 0), static_cast<T>(i));
        }

        // The rest is still 0
        for (int j = 1; j < m.cols(); ++j) {
            for (int i = 0; i < m.rows(); ++i) {
                CHECK_EQ(m(i, j), 0.f);
            }
        }

        CHECK_THROWS(m.row(m.rows()));
    }
}

TEST_CASE_TEMPLATE("Constructing a Matrix with value", T, float, double)
{
    Matrix<T> m(3, 5, 10.f);

    CHECK_EQ(m.rows(), 3);
    CHECK_EQ(m.cols(), 5);

    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            CHECK_EQ(m(i, j), 10.f);
        }
    }

    THEN("set individual values")
    {
        m(2, 4) = 123;
        CHECK_EQ(m(2, 4), 123);
    }

    THEN("fill with a value")
    {
        m = 1235;
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                CHECK_EQ(m(i, j), 1235.f);
            }
        }
    }

    THEN("reshape it")
    {
        m.reshape(5, 3);

        CHECK_EQ(m.rows(), 5);
        CHECK_EQ(m.cols(), 3);
    }

    THEN("resize it")
    {
        m.resize(7, 5);

        CHECK_EQ(m.rows(), 7);
        CHECK_EQ(m.cols(), 5);
    }
}

TEST_CASE_TEMPLATE("Constructing a Matrix from an initializer list", T, float, double)
{
    Matrix<T> m(3, 5, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});

    CHECK_EQ(m.rows(), 3);
    CHECK_EQ(m.cols(), 5);

    int counter = 0;
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            CHECK_EQ(m(i, j), static_cast<T>(counter));
            ++counter;
        }
    }

    THEN("set individual values")
    {
        m(2, 4) = 123;
        CHECK_EQ(m(2, 4), 123.f);
    }

    THEN("fill with a value")
    {
        m = 1235;
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                CHECK_EQ(m(i, j), 1235.f);
            }
        }
    }

    THEN("reshape it")
    {
        m.reshape(5, 3);

        CHECK_EQ(m.rows(), 5);
        CHECK_EQ(m.cols(), 3);
    }

    THEN("resize it")
    {
        m.resize(7, 5);

        CHECK_EQ(m.rows(), 7);
        CHECK_EQ(m.cols(), 5);
    }

    WHEN("Constructing with a wrongly sized initializer list")
    {
        CHECK_THROWS(Matrix<T>(3, 4, {1, 2, 3, 4}));
    }
}

TEST_CASE_TEMPLATE("Constructing a Matrix from an Eigen Matrix", T, float, double)
{
    GIVEN("A row major Eigen Matrix")
    {
        using EigenRowMajorMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        EigenRowMajorMat eigen(4, 5);
        // clang-format off
        eigen <<  1,  2,  3,  4,  5,
                  6,  7,  8,  9, 10,
                 11, 12, 13, 14, 15,
                 16, 17, 18, 19, 20;
        // clang-format on

        Matrix<T> m(eigen);

        CHECK_EQ(m.rows(), 4);
        CHECK_EQ(m.cols(), 5);

        int counter = 1;
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                CHECK_EQ(m(i, j), static_cast<T>(counter));
                ++counter;
            }
        }

        THEN("Check transpose")
        {
            auto mT = m.transpose();

            CHECK_EQ(mT.rows(), m.cols());
            CHECK_EQ(mT.cols(), m.rows());

            INFO(m.cols(), ", ", m.rows());
            for (int i = 0; i < mT.rows(); ++i) {
                for (int j = 0; j < mT.cols(); ++j) {
                    CHECK_EQ(mT(i, j), m(j, i));
                }
            }
        }
    }

    GIVEN("A column major Eigen Matrix")
    {
        using EigenColMajorMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
        EigenColMajorMat eigen(4, 5);
        // clang-format off
        eigen <<  1,  2,  3,  4,  5,
                  6,  7,  8,  9, 10,
                 11, 12, 13, 14, 15,
                 16, 17, 18, 19, 20;
        // clang-format on

        Matrix<T> m(eigen);

        CHECK_EQ(m.rows(), 4);
        CHECK_EQ(m.cols(), 5);

        int counter = 1;
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                CHECK_EQ(m(i, j), static_cast<T>(counter));
                ++counter;
            }
        }

        THEN("Check transpose")
        {
            auto mT = m.transpose();

            CHECK_EQ(mT.rows(), m.cols());
            CHECK_EQ(mT.cols(), m.rows());

            INFO(m.cols(), ", ", m.rows());
            for (int i = 0; i < mT.rows(); ++i) {
                for (int j = 0; j < mT.cols(); ++j) {
                    CHECK_EQ(mT(i, j), m(j, i));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("Testing Matrix Vector Multiplication", T, float, double)
{
    GIVEN("A small 2x3 matrix")
    {
        Matrix<T> m(2, 3, {1, -1, 3, 0, -3, 1});
        Vector<T> x({2, 1, 0});

        auto res = m * x;

        REQUIRE_EQ(res.size(), m.rows());
        CHECK_EQ(res[0], 1.f);
        CHECK_EQ(res[1], -3.f);

        WHEN("Multiplying a vector with the transpose of the matrix")
        {
            auto mT = m.transpose();
            Vector<T> y({4, 3});

            auto mTy = mT * y;

            REQUIRE_EQ(mTy.size(), mT.rows());
            CHECK_EQ(mTy[0], 4.f);
            CHECK_EQ(mTy[1], -13.f);
            CHECK_EQ(mTy[2], 15.f);
        }
    }

    GIVEN("A 4x3 matrix")
    {
        Matrix<T> m(4, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        Vector<T> x({-2, 1, 0});

        auto res = m * x;

        REQUIRE_EQ(res.size(), 4);
        CHECK_EQ(res[0], 0.f);
        CHECK_EQ(res[1], -3.f);
        CHECK_EQ(res[2], -6.f);
        CHECK_EQ(res[3], -9.f);

        WHEN("Multiplying a vector with the transpose of the matrix")
        {
            auto mT = m.transpose();
            Vector<T> y({4, 3, 7, -3});

            auto mTy = mT * y;

            REQUIRE_EQ(mTy.size(), mT.rows());
            CHECK_EQ(mTy[0], 35.f);
            CHECK_EQ(mTy[1], 46.f);
            CHECK_EQ(mTy[2], 57.f);
        }
    }
}

TEST_SUITE_END();
