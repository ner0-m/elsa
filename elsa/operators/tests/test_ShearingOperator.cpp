/**
 * @file test_ShearingOperator.cpp
 *
 * @brief Tests for the ShearingOperator class
 *
 * @author Andi Braimllari
 */

#include "ShearingOperator.h"
#include "VolumeDescriptor.h"

#include <catch2/catch.hpp>

using namespace elsa;

SCENARIO("Using a ShearingOperator")
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 2, 8;
        VolumeDescriptor dd(numCoeff);
        DataContainer<real_t> x(dd);
        x = 3;

        WHEN("applying isotropic scaling")
        {
            real_t shearingParameter{4};
            ShearingOperator<real_t> shearingOp(dd, shearingParameter);

            THEN("apply and applyAdjoint yield the correct results")
            {
                auto shearedData = shearingOp.apply(x);
                for (index_t i = 0; i < shearedData.getSize(); ++i)
                    printf("value is %f\n", shearedData[i]);
            }
        }
    }
}
