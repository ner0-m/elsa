#include "doctest/doctest.h"

#include "CGLS.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "Identity.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("CGLS: Solving a simple linear problem", data_t, float, double)
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

        CGLS<data_t> cgls(id, b, 0);

        auto result = cgls.solve(1);

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

        CGLS<data_t> cgls(id, b, 0);

        auto result = cgls.solve(1);

        CAPTURE(b);
        CAPTURE(result);
        for (int i = 0; i < result.getSize(); ++i) {
            CAPTURE(i);
            CHECK(checkApproxEq(result[i], bVec[i]));
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

        Scaling<data_t> scale{dd, 5};

        CGLS<data_t> cgls(scale, b, 0);

        auto result = cgls.solve(1);

        CAPTURE(bVec.transpose());
        CAPTURE(result);
        for (int i = 0; i < result.getSize(); ++i) {
            CAPTURE(i);
            CHECK_EQ(5. * result[i], doctest::Approx(bVec[i]));
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

        Scaling<data_t> scale{dd, 0.5};

        CGLS<data_t> cgls(scale, b, 0);

        auto result = cgls.solve(1);

        CAPTURE(b);
        CAPTURE(result);
        for (int i = 0; i < result.getSize(); ++i) {
            CAPTURE(i);
            CHECK_EQ(0.5 * result[i], doctest::Approx(bVec[i]));
        }
    }
}

TEST_SUITE_END();
