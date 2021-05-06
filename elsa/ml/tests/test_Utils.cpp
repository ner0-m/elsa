/**
 * @file test_common.cpp
 *
 * @brief Tests for common ml functionality
 *
 * @author David Tellenbach
 */

#include "doctest/doctest.h"

#include "elsaDefines.h"
#include "DataContainer.h"
#include "VolumeDescriptor.h"
#include "Utils.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("ml");

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_CASE("Encoding")
{
    SECTION("Encode one-hot")
    {
        index_t numClasses = 10;
        index_t batchSize = 4;
        IndexVector_t dims{{batchSize}};
        VolumeDescriptor desc(dims);

        // For each entry in a batch we require one label
        Eigen::VectorXf data{{3, 0, 9, 1}};

        /* This should give the following one-hot encoding
        {
      Idx  0  1  2  3  4  5  6  7  8  9
          {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
          {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
          {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        }
        */

        DataContainer<real_t> notOneHot(desc, data);
        auto oneHot = ml::Utils::Encoding::toOneHot(notOneHot, numClasses, batchSize);

        REQUIRE(oneHot[3 + 0 * numClasses] == 1.f);
        REQUIRE(oneHot[0 + 1 * numClasses] == 1.f);
        REQUIRE(oneHot[9 + 2 * numClasses] == 1.f);
        REQUIRE(oneHot[1 + 3 * numClasses] == 1.f);
    }

    SECTION("Decode one-hot")
    {
        index_t numClasses = 10;
        IndexVector_t dims{{numClasses, 4}};
        VolumeDescriptor desc(dims);
        Eigen::VectorXf data{{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}};
        DataContainer<real_t> dc(desc, data);
        auto fromOneHot = ml::Utils::Encoding::fromOneHot(dc, numClasses);
        REQUIRE(fromOneHot[0] == Approx(3.f));
        REQUIRE(fromOneHot[1] == Approx(0.f));
        REQUIRE(fromOneHot[2] == Approx(9.f));
        REQUIRE(fromOneHot[3] == Approx(1.f));
    }
}
TEST_SUITE_END();
