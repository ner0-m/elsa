#include "doctest/doctest.h"

#include "CGNE.h"
#include "Logger.h"
#include "Scaling.h"
#include "VolumeDescriptor.h"
#include "Identity.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("CGNE: Solving a simple linear problem", data_t, float, double)
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);
    srand((unsigned int) 666);

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 10;
        VolumeDescriptor dd{numCoeff};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        Identity<data_t> id{dd};

        CGNE<data_t> cgne(id, b);

        auto result = cgne.solve(1);

        CAPTURE(bVec.transpose());
        CAPTURE(result);
        for (int i = 0; i < result.getSize(); ++i) {
            CAPTURE(i);
            CHECK(checkApproxEq(result[i], bVec[i]));
        }
    }

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 10, 15;
        VolumeDescriptor dd{numCoeff};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        Identity<data_t> id{dd};

        CGNE<data_t> cgne(id, b);

        auto result = cgne.solve(1);

        CAPTURE(b);
        CAPTURE(result);
        for (int i = 0; i < result.getSize(); ++i) {
            CAPTURE(i);
            CHECK(checkApproxEq(result[i], bVec[i]));
        }

        THEN("A clone is equal to the original")
        {
            auto clone = cgne.clone();

            CHECK_EQ(*clone, cgne);
        }
    }

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 10;
        VolumeDescriptor dd{numCoeff};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        Scaling<data_t> scaling{dd, 5};

        CGNE<data_t> cgne(scaling, b);

        auto result = cgne.solve(1);

        CAPTURE(bVec.transpose());
        CAPTURE(result);
        for (int i = 0; i < result.getSize(); ++i) {
            CAPTURE(i);
            CHECK_EQ(5. * result[i], doctest::Approx(bVec[i]));
        }

        THEN("A clone is equal to the original")
        {
            auto clone = cgne.clone();

            CHECK_EQ(*clone, cgne);
        }
    }

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 10, 15;
        VolumeDescriptor dd{numCoeff};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        Scaling<data_t> scaling{dd, 0.5};

        CGNE<data_t> cgne(scaling, b);

        auto result = cgne.solve(1);

        CAPTURE(b);
        CAPTURE(result);
        for (int i = 0; i < result.getSize(); ++i) {
            CAPTURE(i);
            CHECK_EQ(0.5 * result[i], doctest::Approx(bVec[i]));
        }

        THEN("A clone is equal to the original")
        {
            auto clone = cgne.clone();

            CHECK_EQ(*clone, cgne);
        }
    }
}

TEST_SUITE_END();

