/**
 * \file test_FourierTransform.cpp
 *
 * \brief Tests for the fourier transform operator
 *
 * \author Jonas Jelten
 */

#include <catch2/catch.hpp>
#include "FourierTransform.h"
#include "VolumeDescriptor.h"

using namespace elsa;

SCENARIO("Constructing a FourierTransform operator")
{
    GIVEN("a descriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 45, 28;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating a FourierTransform operator")
        {
            FourierTransform fftOp(dd);

            THEN("the DataDescriptors are as expected")
            {
                REQUIRE(fftOp.getDomainDescriptor() == dd);
                REQUIRE(fftOp.getRangeDescriptor() == dd);
            }
        }

        WHEN("cloning a FourierTransform operator")
        {
            FourierTransform fftOp(dd);
            auto fftOpClone = fftOp.clone();

            THEN("everything matches")
            {
                REQUIRE(fftOpClone.get() != &fftOp);
                REQUIRE(*fftOpClone == fftOp);
            }
        }
    }
}

SCENARIO("Using FourierTransform")
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 13;
        VolumeDescriptor dd(numCoeff);
        DataContainer<complex_t> input(dd);
        input = 3.3f;

        FourierTransform fftOp(dd);

        WHEN("applying the fft")
        {
            auto output = fftOp.apply(input);

            THEN("the result is as expected")
            {
                REQUIRE(output.getSize() == input.getSize());
                //REQUIRE(input == output);
            }
        }

        WHEN("applying the adjoint of fft")
        {
            auto output = fftOp.applyAdjoint(input);

            THEN("the results is as expected")
            {
                REQUIRE(output.getSize() == input.getSize());
                //REQUIRE(input == output);
            }
        }

        WHEN("applying the fft and inverse fft")
        {
            auto output = fftOp.apply(fftOp.applyAdjoint(input));

            THEN("the results is as expected")
            {
                    REQUIRE(output.getSize() == input.getSize());
                    REQUIRE(input == output);
            }
        }
    }
}
