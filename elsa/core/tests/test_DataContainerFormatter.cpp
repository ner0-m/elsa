#include "doctest/doctest.h"
#include "DataContainerFormatter.hpp"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE("DataContainerFormatter: default config")
{
    using data_t = float;

    std::stringstream buffer;
    DataContainerFormatter<data_t> formatter;

    GIVEN("A 1D DataContainer with arbitrary values")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 10;

        DataContainer<data_t> dc(VolumeDescriptor{numCoeff});
        dc = 1;

        THEN("Formatting writes correctly to the stream")
        {
            formatter.format(buffer, dc);

            auto resultString = buffer.str();
            REQUIRE_EQ(resultString,
                       "DataContainer<dims=1, shape=(10)>\n[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]");
        }
    }

    GIVEN("A larger 1D DataContainer with arbitrary values")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 15;

        DataContainer<data_t> dc(VolumeDescriptor{numCoeff});
        dc = 1;

        THEN("Formatting writes abbreviated to the stream")
        {
            formatter.format(buffer, dc);

            auto resultString = buffer.str();
            REQUIRE_EQ(
                resultString,
                "DataContainer<dims=1, shape=(15)>\n[1, 1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1, 1]");
        }
    }

    GIVEN("A 2D DataContainer with arbitrary values")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 5, 3;

        DataContainer<data_t> dc(VolumeDescriptor{numCoeff});
        dc = 1;

        THEN("Formatting writes correctly to the stream")
        {
            formatter.format(buffer, dc);

            auto resultString = buffer.str();
            REQUIRE_EQ(resultString,
                       R"(DataContainer<dims=2, shape=(5 3)>
[[1, 1, 1],
 [1, 1, 1],
 [1, 1, 1],
 [1, 1, 1],
 [1, 1, 1]])");
        }
    }
}

TEST_SUITE_END();
