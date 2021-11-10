/**
 * @file test_FourierTransform.cpp
 *
 * @brief Tests for the fourier transform operator
 *
 * @author Jonas Jelten
 */

#include "ExpressionPredicates.h"
#include "doctest/doctest.h"
#include "elsaDefines.h"
#include "testHelpers.h"
#include "FourierTransform.h"
#include "VolumeDescriptor.h"
#include "DataHandlerCPU.h"
#include <type_traits>

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

TEST_CASE_TEMPLATE("FourierTransform: Testing 1d", data_t, float, double)
{
    GIVEN("a volume descriptor")
    {
        IndexVector_t numCoeff{1};
        numCoeff << 4;
        VolumeDescriptor dd{numCoeff};

        WHEN("performing 1d fft transform")
        {
            using namespace std::complex_literals;
            FourierTransform<complex<data_t>> fftOp{dd};

            Vector_t<complex<data_t>> input{4};

            // TODO: somehow magically do complex conversions with eigens , operator
            if constexpr (std::is_same_v<data_t, float>) {
                input << 0.f + 0.if, 42.f + 0.if, 42.f + 0.if, 0.f + 0.if;
            } else {
                input << 0. + 0.i, 42. + 0.i, 42. + 0.i, 0. + 0.i;
            }
            Vector_t<complex<data_t>> expected{4};

            // TODO: also check other datahandlers

            THEN("the forward transformed values are correct")
            {
                DataContainer<complex<data_t>> inputdc{dd, input, DataHandlerType::CPU};
                auto output = fftOp.apply(inputdc);

                if constexpr (std::is_same_v<data_t, float>) {
                    expected << 84.f + 0.if, -42.f - 42.if, 0.f + 0.if, -42.f + 42.if;
                } else {
                    expected << 84. + 0.i, -42. - 42.i, 0. + 0.i, -42. + 42.i;
                }

                for (index_t i = 0; i < output.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(output[i], expected(i)));
                }
            }

            THEN("the inverse transformed values are correct")
            {
                DataContainer<complex<data_t>> inputdc{dd, input, DataHandlerType::CPU};
                auto output = fftOp.applyAdjoint(inputdc);

                if constexpr (std::is_same_v<data_t, float>) {
                    expected << 21.f + 0.if, -10.5f + 10.5if, 0.f + 0.if, -10.5f - 10.5if;
                } else {
                    expected << 21. + 0.i, -10.5 + 10.5i, 0. + 0.i, -10.5 - 10.5i;
                }

                for (index_t i = 0; i < output.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(output[i], expected(i)));
                }
            }

            THEN("the forward-inverse transformed values are correct")
            {
                DataContainer<complex<data_t>> inputdc{dd, input, DataHandlerType::CPU};

                auto output = fftOp.applyAdjoint(fftOp.apply(inputdc));

                if constexpr (std::is_same_v<data_t, float>) {
                    expected << 0.f + 0.if, 42.f + 0.if, 42.f + 0.if, 0.f + 0.if;
                } else {
                    expected << 0. + 0.i, 42. + 0.i, 42. + 0.i, 0. + 0.i;
                }

                for (index_t i = 0; i < output.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(output[i], expected(i)));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("FourierTransform: 2d test", data_t, float, double)
{
    using namespace std::complex_literals;

    GIVEN("some data")
    {

        WHEN("applying the fft")
        {
            IndexVector_t size{2};
            size << 4, 4;
            VolumeDescriptor dd{size};

            FourierTransform<complex<data_t>> fftOp{dd};

            DataContainer<data_t> testdata{dd};
            testdata = 0.0;
            testdata(2, 2) = 42.0f;
            testdata(0, 2) = 42.0f;

            auto input = testdata.asComplex();

            auto output = fftOp.apply(input);

            Vector_t<complex<data_t>> expected{4 * 4};
            // transposed because vector!
            expected << 84, 0, 84, 0, -84, 0, -84, 0, 84, 0, 84, 0, -84, 0, -84, 0;

            THEN("the result is as expected")
            {
                REQUIRE(output.getSize() == input.getSize());
                for (index_t i = 0; i < output.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(output[i], expected(i)));
                }
            }
        }

        WHEN("applying the adjoint of fft")
        {
            IndexVector_t size{2};
            size << 4, 4;
            VolumeDescriptor dd{size};

            FourierTransform<complex<data_t>> fftOp{dd};

            DataContainer<data_t> testdata{dd};
            testdata = 0.0;
            testdata(2, 2) = 42.0f;
            testdata(0, 2) = 42.0f;

            auto input = testdata.asComplex();

            auto output = fftOp.apply(input);

            Vector_t<complex<data_t>> expected{4 * 4};
            // transposed because vector!
            expected << 84, 0, 84, 0, -84, 0, -84, 0, 84, 0, 84, 0, -84, 0, -84, 0;

            THEN("the result is as expected")
            {
                REQUIRE(output.getSize() == input.getSize());
                for (index_t i = 0; i < output.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(output[i], expected(i)));
                }
            }
        }

        WHEN("applying the fft and inverse fft")
        {

            IndexVector_t size{4};
            size << 5, 10, 15, 20;
            VolumeDescriptor dd{size};
            FourierTransform<complex<data_t>> fftOp{dd};

            auto [input, randVec] =
                generateRandomContainer<complex<data_t>>(dd, DataHandlerType::CPU);

            auto mid = fftOp.apply(input);
            auto output = fftOp.applyAdjoint(mid);

            THEN("the back and forward fourier transform result in the original values")
            {
                REQUIRE(output.getSize() == input.getSize());
                for (index_t i = 0; i < output.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(output[i], randVec(i)));
                }
            }
        }

        WHEN("ensuring the linearity of a fft")
        {

            IndexVector_t size{2};
            size << 20, 10;
            VolumeDescriptor dd{size};

            auto [inputA, randVecA] =
                generateRandomContainer<complex<data_t>>(dd, DataHandlerType::CPU);
            auto [inputB, randVecB] =
                generateRandomContainer<complex<data_t>>(dd, DataHandlerType::CPU);

            FourierTransform<complex<data_t>> fftOp{dd};

            auto fftA = fftOp.apply(inputA);
            auto fftB = fftOp.apply(inputB);

            THEN("fft(A + B) = fft(A) + fft(B)")
            {
                DataContainer<complex<data_t>> fftA_plus_fftB = fftA + fftB;
                DataContainer<complex<data_t>> A_plus_B = inputA + inputB;

                auto fftAB = fftOp.apply(A_plus_B);

                REQUIRE(fftAB.getSize() == fftA_plus_fftB.getSize());
                for (index_t i = 0; i < fftAB.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(fftA_plus_fftB[i], fftAB[i]));
                }
            }

            THEN("f(x * A) = x * f(A)")
            {
                complex<data_t> x = 42;
                DataContainer<complex<data_t>> fftA_scaled = (fftA * x);

                auto fftAx = fftOp.apply(inputA * x);

                REQUIRE(fftA_scaled.getSize() == fftAx.getSize());
                for (index_t i = 0; i < fftA_scaled.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(fftA_scaled[i], fftAx[i]));
                }
            }
        }

        // TODO impulse function response
    }
}
TEST_SUITE_END();
