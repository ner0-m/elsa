#include "doctest/doctest.h"
#include "NoiseGenerators.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("generators");

TEST_CASE_TEMPLATE("Noise generators:", data_t, float, double)
{
    GIVEN("A random data container")
    {
        auto [dc, mat] =
            generateRandomContainer<data_t>(VolumeDescriptor({32, 32}), DataHandlerType::CPU);

        WHEN("Adding no noise (NoNoiseGenerator)")
        {
            auto generator = NoNoiseGenerator{};
            auto not_noisy = generator(dc);

            THEN("Nothing happens")
            {
                CHECK_UNARY(isCwiseApprox(dc, not_noisy));
            }
        }

        WHEN("Adding Gaussian noise (GaussianNoiseGenerator)")
        {
            auto generator = GaussianNoiseGenerator{0, 0.25};
            auto noisy = generator(dc);

            // TODO: Find a way to properly test this!
            THEN("Something happens")
            {
                CHECK_UNARY_FALSE(isCwiseApprox(dc, noisy));
            }
        }

        WHEN("Adding Poisson noise (PoissonNoiseGenerator)")
        {
            auto generator = PoissonNoiseGenerator{0};
            auto noisy = generator(dc);

            // TODO: Find a way to properly test this!
            THEN("Something happens")
            {
                CHECK_UNARY_FALSE(isCwiseApprox(dc, noisy));
            }
        }
    }
}

TEST_SUITE_END();
