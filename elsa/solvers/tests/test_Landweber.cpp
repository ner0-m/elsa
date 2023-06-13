#include "doctest/doctest.h"

#include "Landweber.h"
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

TEST_CASE_TEMPLATE("Landweber: Solving a simple linear problem", data_t, float, double)
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

    GIVEN("An operator and data")
    {
        Landweber<data_t> solver(A, b, 1);

        auto x = solver.solve(1);

        DataContainer<data_t> zero{desc};
        zero = 0;

        auto expected = -A.applyAdjoint(A.apply(zero) - b);

        for (index_t i = 0; i < desc.getNumberOfCoefficients(); ++i) {
            CHECK_EQ(x[i], expected[i]);
        }

        THEN("Check clone is equal")
        {
            auto clone = solver.clone();

            CHECK_EQ(*clone, solver);
        }
    }
}
TEST_SUITE_END();
