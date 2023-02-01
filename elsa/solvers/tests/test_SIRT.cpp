#include "doctest/doctest.h"

#include "SIRT.h"
#include "Scaling.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("SIRT: Solving a simple linear problem", data_t, float, double)
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    IndexVector_t numCoeff(2);
    numCoeff << 16, 16;
    VolumeDescriptor desc{numCoeff};

    // Setup b
    Vector_t<data_t> vec = Vector_t<data_t>::Random(desc.getNumberOfCoefficients());
    DataContainer<data_t> b{desc, vec};

    // Setup A
    Scaling<data_t> A{desc, 5.f};

    GIVEN("A WLSProblem")
    {
        WLSProblem<data_t> prob{A, b};
        SIRT<data_t> solver(prob);

        auto x = solver.solve(1);

        DataContainer<data_t> zero{desc};
        zero = 0;

        auto T = Scaling<data_t>{desc, 1 / 5.f};
        auto M = Scaling<data_t>{desc, 1 / 5.f};

        auto op = T * adjoint(A) * M;

        auto expected = -op.apply(A.apply(zero) - b);

        for (index_t i = 0; i < desc.getNumberOfCoefficients(); ++i) {
            CHECK_EQ(x[i], doctest::Approx(expected[i]));
        }

        THEN("Check clone is equal")
        {
            auto clone = solver.clone();

            CHECK_EQ(*clone, solver);
        }
    }

    GIVEN("A WLSProblem with step size")
    {
        WLSProblem<data_t> prob{A, b};
        SIRT<data_t> solver(prob, 0.5f);

        auto x = solver.solve(1);

        DataContainer<data_t> zero{desc};
        zero = 0;

        auto T = Scaling<data_t>{desc, 1 / 5.f};
        auto M = Scaling<data_t>{desc, 1 / 5.f};

        auto op = T * adjoint(A) * M;

        auto expected = -0.5f * op.apply(A.apply(zero) - b);

        for (index_t i = 0; i < desc.getNumberOfCoefficients(); ++i) {
            CHECK_EQ(x[i], doctest::Approx(expected[i]));
        }

        THEN("Check clone is equal")
        {
            auto clone = solver.clone();

            CHECK_EQ(*clone, solver);
        }
    }

    GIVEN("An operator and data")
    {
        SIRT<data_t> solver(A, b);

        auto x = solver.solve(1);

        DataContainer<data_t> zero{desc};
        zero = 0;

        auto T = Scaling<data_t>{desc, 1 / 5.f};
        auto M = Scaling<data_t>{desc, 1 / 5.f};

        auto op = T * adjoint(A) * M;

        auto expected = -op.apply(A.apply(zero) - b);

        for (index_t i = 0; i < desc.getNumberOfCoefficients(); ++i) {
            CHECK_EQ(x[i], doctest::Approx(expected[i]));
        }

        THEN("Check clone is equal")
        {
            auto clone = solver.clone();

            CHECK_EQ(*clone, solver);
        }
    }
}
TEST_SUITE_END();
