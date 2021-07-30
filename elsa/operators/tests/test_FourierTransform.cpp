/**
 * \file test_FourierTransform.cpp
 *
 * \brief Tests for the fourier transform operator
 *
 * \author Jonas Jelten
 */

#include "doctest/doctest.h"
#include "FourierTransform.h"
#include "VolumeDescriptor.h"

using namespace elsa;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("FourierTransform: Testing construction", data_t, float, double)
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

TEST_CASE_TEMPLATE("FourierTransform: 2d test", data_t, float, double)
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 2, 4;
        VolumeDescriptor dd(numCoeff);
        using cdata_t = std::complex<data_t>;

        FourierTransform<std::complex<data_t>> fftOp{dd};

        WHEN("applying the fft")
        {
            Vector_t<data_t>
            DataContainer<std::complex<data_t>> input{dd};
            input = 3.3f;
            auto output = fftOp.apply(input);

            THEN("the result is as expected")
            {
                REQUIRE(output2.getSize() == input.getSize());
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

        // TODO ensure identity
        // TODO ensure linearity
        // TODO impulse function response
    }
}
