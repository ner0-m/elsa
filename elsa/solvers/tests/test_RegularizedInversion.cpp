#include "DataContainer.h"
#include "doctest/doctest.h"

#include "RegularizedInversion.h"
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

TEST_CASE_TEMPLATE("RegularizedInversion: Solving a simple linear problem", data_t, float, double)
{
    // GIVEN("a linear problem")
    // {
    //     IndexVector_t numCoeff(1);
    //     numCoeff << 10;
    //     VolumeDescriptor dd{numCoeff};
    //
    //     Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
    //     bVec.setRandom();
    //     DataContainer<data_t> b{dd, bVec};
    //
    //     Scaling<data_t> scale{dd, 4};
    //
    //     Identity<data_t> id{dd};
    //     Vector_t<data_t> tVec(dd.getNumberOfCoefficients());
    //     tVec.setZero();
    //     DataContainer<data_t> t(dd, tVec);
    //
    //     auto result = reguarlizedInversion(scale, b, id, t, 1, {}, 1);
    //
    //     // Reduce it just a bit, to compensate for the L2 regularization
    //     auto expected = b / 4.3;
    //
    //     CAPTURE(expected);
    //     CAPTURE(result);
    //     CHECK_EQ(expected.getSize(), result.getSize());
    //     for (int i = 0; i < result.getSize(); ++i) {
    //         CAPTURE(i);
    //         CHECK(checkApproxEq(result[i], expected[i]));
    //     }
    // }

    GIVEN("a weighted linear problem")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 10;
        VolumeDescriptor dd{numCoeff};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        Scaling<data_t> scale{dd, 1};

        Identity<data_t> id{dd};
        Vector_t<data_t> tVec(dd.getNumberOfCoefficients());
        tVec.setZero();
        DataContainer<data_t> t(dd, tVec);

        Vector_t<data_t> wVec(dd.getNumberOfCoefficients());
        wVec.setRandom();
        // DataContainer<data_t> w{dd, 1. / wVec.array()};
        DataContainer<data_t> w{dd, wVec.array()};

        auto result = reguarlizedInversion(scale, b, id, t, 1, std::make_optional(w), 1);

        // Compensate for L2 Regularization
        auto expected = Scaling<data_t>(b.getDataDescriptor(), w).apply(b * 4);

        CAPTURE(w);
        CAPTURE(expected);
        CAPTURE(result);
        CHECK_EQ(expected.getSize(), result.getSize());
        for (int i = 0; i < result.getSize(); ++i) {
            CAPTURE(i);
            CHECK(checkApproxEq(result[i], expected[i]));
        }
    }
}

TEST_SUITE_END();
