#include "doctest/doctest.h"

#include "Math.hpp"
#include "VolumeDescriptor.h"

#include <testHelpers.h>

TEST_SUITE_BEGIN("Math");

using namespace elsa;

TEST_CASE("Math::factorial")
{
    CHECK_EQ(1, math::factorial(0));
    CHECK_EQ(1, math::factorial(1));

    auto fac = 1;
    for (int i = 1; i < 10; ++i) {
        fac *= i;
        CHECK_EQ(fac, math::factorial(i));
    }
}

TEST_CASE("Math::binom")
{
    GIVEN("n == 10")
    {
        const index_t n = 10;

        WHEN("k larger than n") { CHECK_EQ(math::binom(n, 15), 0); }

        WHEN("k == 0 or k == n")
        {
            CHECK_EQ(math::binom(n, 0), 1);
            CHECK_EQ(math::binom(n, 10), 1);
        }

        CHECK_EQ(math::binom(n, 1), 10);
        CHECK_EQ(math::binom(n, 2), 45);
        CHECK_EQ(math::binom(n, 3), 120);
        CHECK_EQ(math::binom(n, 4), 210);
        CHECK_EQ(math::binom(n, 5), 252);
        CHECK_EQ(math::binom(n, 6), 210);
        CHECK_EQ(math::binom(n, 7), 120);
        CHECK_EQ(math::binom(n, 8), 45);
        CHECK_EQ(math::binom(n, 9), 10);
    }
}

TEST_CASE("Math::heaviside")
{
    constexpr index_t size = 200;
    constexpr real_t c = 0.5;
    const auto linspace = Vector_t<real_t>::LinSpaced(size, -2, 2);

    for (std::size_t i = 0; i < size; ++i) {
        auto res = math::heaviside(linspace[i], c);
        if (linspace[i] == 0.) {
            CHECK_EQ(res, c);
        } else if (linspace[i] < 0) {
            CHECK_EQ(res, 0);
        } else if (linspace[i] > 0) {
            CHECK_EQ(res, 1);
        }
    }
}

TEST_CASE_TEMPLATE("Math: Testing the statistics", TestType, float, double)
{
    GIVEN("some DataContainers")
    {
        IndexVector_t sizeVector(2);
        sizeVector << 2, 4;
        VolumeDescriptor volDescr(sizeVector);

        Vector_t<TestType> vect1(volDescr.getNumberOfCoefficients());
        vect1 << 4, 2, 0.7f, 1, 0, 9, 53, 8;

        Vector_t<TestType> vect2(volDescr.getNumberOfCoefficients());
        vect2 << 5, 1, 6, 12, 22, 23, 9, 9;

        DataContainer<TestType> dataCont1(volDescr, vect1);

        DataContainer<TestType> dataCont2(volDescr, vect2);

        WHEN("running the Mean Squared Error")
        {
            DataContainer<TestType> dataCont3(VolumeDescriptor{{4, 3, 8}});
            THEN("it throws if the containers have different shapes")
            {
                REQUIRE_THROWS_AS(statistics::meanSquaredError<TestType>(dataCont1, dataCont3),
                                  InvalidArgumentError);
            }

            THEN("it produces the correct result")
            {
                auto meanSqErr = statistics::meanSquaredError<TestType>(dataCont1, dataCont2);
                REQUIRE_UNARY(checkApproxEq(meanSqErr, 346.01125f));
            }
        }

        WHEN("running the Relative Error")
        {
            DataContainer<TestType> dataCont3(VolumeDescriptor{{7, 6, 5, 4}});
            THEN("it throws if the containers have different shapes")
            {
                REQUIRE_THROWS_AS(statistics::relativeError<TestType>(dataCont1, dataCont3),
                                  InvalidArgumentError);
            }

            THEN("it produces the correct result")
            {
                auto relErr = statistics::relativeError<TestType>(dataCont1, dataCont2);
                REQUIRE_UNARY(checkApproxEq(relErr, 1.4157718f));
            }
        }

        WHEN("running the Peak Signal-to-Noise Ratio")
        {
            DataContainer<TestType> dataCont3(VolumeDescriptor{4});
            THEN("it throws if the containers have different shapes")
            {
                REQUIRE_THROWS_AS(
                    statistics::peakSignalToNoiseRatio<TestType>(dataCont1, dataCont3),
                    InvalidArgumentError);
            }

            auto psnr = statistics::peakSignalToNoiseRatio<TestType>(dataCont1, dataCont2);
            auto expectedpsnr = static_cast<TestType>(9.09461);
            THEN("it produces the correct result")
            {
                REQUIRE_UNARY(checkApproxEq(psnr, expectedpsnr));
            }
        }
    }

    GIVEN("some vectors of numbers")
    {

        std::vector<TestType> numbers = {5, 0, 14, -4, 8};
        std::vector<TestType> manyNumbers = {2, 6,  9, 4,  2, 0,  7, 9,  4, 8, 6, 6,
                                             9, 4,  2, 15, 6, 91, 4, 22, 2, 3, 9, 4,
                                             1, -2, 6, 9,  4, 2,  5, 6,  4, 4, 9};

        WHEN("calculating the mean, standard deviation and a 95% confidence interval")
        {
            auto [mean, stddev] = statistics::calculateMeanStddev(numbers);

            THEN("it produces the correct mean and standard deviation")
            {
                TestType expectedmean = 4.6f;
                TestType expectedstddev = 6.2481997f;

                REQUIRE_UNARY(checkApproxEq(mean, expectedmean));
                REQUIRE_UNARY(checkApproxEq(stddev, expectedstddev));
            }

            auto [lower, upper] = statistics::confidenceInterval95(numbers.size(), mean, stddev);

            THEN("it produces the expected 95% confidence interval")
            {
                float expectedlower = -12.74778f;
                float expectedupper = 21.94778f;

                REQUIRE_UNARY(checkApproxEq(lower, expectedlower));
                REQUIRE_UNARY(checkApproxEq(upper, expectedupper));
            }
        }

        WHEN("calculating the mean, standard deviation and a 95% confidence interval for more than "
             "30 numbers")
        {
            auto [mean, stddev] = statistics::calculateMeanStddev(manyNumbers);

            THEN("it produces the correct mean and standard deviation")
            {
                TestType expectedmean = 8.0571429f;
                TestType expectedstddev = 14.851537f;

                REQUIRE_UNARY(checkApproxEq(mean, expectedmean));
                REQUIRE_UNARY(checkApproxEq(stddev, expectedstddev));
            }

            auto [lower, upper] =
                statistics::confidenceInterval95(manyNumbers.size(), mean, stddev);

            THEN("it produces the expected 95% confidence interval")
            {
                float expectedlower = -21.05187f;
                float expectedupper = 37.16616f;

                REQUIRE_UNARY(checkApproxEq(lower, expectedlower));
                REQUIRE_UNARY(checkApproxEq(upper, expectedupper));
            }
        }
    }
}

TEST_SUITE_END();
